"""Step 6 entry point: bench-test and calibrate the VESC (Flipsky 75100 Pro V2).

    python -m car.bench --port /dev/ttyACM0

Interactive menu:

    1  telemetry        battery voltage, ERPM, currents, temperatures
    2  motor spin test  gentle duty ramp -- WHEELS OFF THE GROUND
    3  steering trim    find servo center / full-left / full-right
    4  drive ratio      compute ERPM-per-(m/s) from wheel + gearing numbers
    5  keyboard teleop  WASD with an automatic dead-man stop
    6  save config      writes car_link.yaml for the autonomy stack
    q  quit (motor stopped)

Every motor command in this tool is capped to gentle limits on purpose.
"""

from __future__ import annotations

import argparse
import math
import select
import sys
import termios
import time
import tty
from pathlib import Path

import yaml

from .vesc_uart import VescUART

MAX_BENCH_DUTY = 0.10          # spin test cap
MAX_TELEOP_CURRENT_A = 12.0    # drive torque cap during teleop
MAX_TELEOP_BRAKE_A = 25.0
MIN_VOLTAGE_WARN = 14.0        # the Flipsky 75100 needs >= 14 V (4S and up)


# --------------------------------------------------------------- tty helpers

class RawKeys:
    """Non-blocking single-key reads, restores the terminal on exit."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def get(self, timeout_s: float = 0.0) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        return sys.stdin.read(1) if ready else None


def _prompt_float(text: str, lo: float, hi: float) -> float:
    while True:
        try:
            v = float(input(text).strip())
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  !! enter a number between {lo} and {hi}")


# ------------------------------------------------------------------- modes

def telemetry(vesc: VescUART) -> None:
    print("streaming telemetry, Ctrl-C to stop:")
    try:
        while True:
            v = vesc.get_values()
            if v is None:
                print("  (no response)")
            else:
                warn = "  << LOW! 75100 needs >=14 V" if v.v_in < MIN_VOLTAGE_WARN else ""
                print(f"  battery {v.v_in:5.1f} V{warn} | erpm {v.erpm:8.0f} | "
                      f"motor {v.motor_current_a:5.1f} A | "
                      f"input {v.input_current_a:5.1f} A | duty {v.duty:+.2f} | "
                      f"fet {v.temp_fet_c:4.1f} C | fault {v.fault}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print()


def spin_test(vesc: VescUART) -> None:
    if input("Type WHEELS OFF to confirm the car is on a stand: ").strip() \
            != "WHEELS OFF":
        print("aborted.")
        return
    print(f"ramping duty 0 -> {MAX_BENCH_DUTY:.0%} -> 0 over ~6 s")
    steps = 30
    try:
        for i in list(range(steps + 1)) + list(range(steps, -1, -1)):
            duty = MAX_BENCH_DUTY * i / steps
            vesc.set_duty(duty)
            if i % 10 == 0:
                v = vesc.get_values()
                if v:
                    print(f"  duty {duty:+.3f} -> erpm {v.erpm:8.0f}  "
                          f"motor {v.motor_current_a:4.1f} A")
            time.sleep(0.1)
    finally:
        vesc.set_current(0.0)
    print("spin test done.")


def steering_trim(vesc: VescUART, cfg: dict) -> None:
    pos = cfg.get("servo_center", 0.5)
    print("keys: a/d = nudge left/right, A/D = big nudge, c = save center, "
          "l = save full-left, r = save full-right, q = done")
    with RawKeys() as keys:
        while True:
            vesc.set_servo(pos)
            k = keys.get(0.05)
            if k is None:
                continue
            if k == "a":
                pos = max(pos - 0.005, 0.1)
            elif k == "d":
                pos = min(pos + 0.005, 0.9)
            elif k == "A":
                pos = max(pos - 0.03, 0.1)
            elif k == "D":
                pos = min(pos + 0.03, 0.9)
            elif k == "c":
                cfg["servo_center"] = round(pos, 3)
                print(f"\r  center = {pos:.3f}")
            elif k == "l":
                cfg["servo_left"] = round(pos, 3)
                print(f"\r  full-left = {pos:.3f}")
            elif k == "r":
                cfg["servo_right"] = round(pos, 3)
                print(f"\r  full-right = {pos:.3f}")
            elif k == "q":
                break
            print(f"\r  servo {pos:.3f}   ", end="", flush=True)
    vesc.set_servo(cfg.get("servo_center", 0.5))
    print()


def drive_ratio(cfg: dict) -> None:
    print("ERPM = wheel_speed * 60 * gear_ratio * pole_pairs / circumference")
    d = _prompt_float("  wheel diameter [m] (e.g. 0.085): ", 0.02, 0.5)
    g = _prompt_float("  total gear ratio motor:wheel (e.g. 8.0): ", 1.0, 50.0)
    pp = _prompt_float("  motor pole pairs (Xerun 3660 G3 = 2): ", 1, 30)
    erpm_per_mps = 60.0 * g * pp / (math.pi * d)
    cfg.update(wheel_diameter_m=d, gear_ratio=g, motor_pole_pairs=int(pp),
               erpm_per_mps=round(erpm_per_mps, 1))
    print(f"  -> {erpm_per_mps:.0f} ERPM per m/s "
          f"(e.g. 5 m/s = {5 * erpm_per_mps:.0f} ERPM)")


def teleop(vesc: VescUART, cfg: dict) -> None:
    center = cfg.get("servo_center", 0.5)
    left = cfg.get("servo_left", 0.2)
    right = cfg.get("servo_right", 0.8)
    print("teleop: w/s = drive/brake, a/d = steer, space = stop, q = quit.")
    print("DEAD-MAN: releasing the keys stops the car within 0.4 s.")
    steer = 0.0                          # -1 (left) .. +1 (right)
    last_drive = 0.0
    with RawKeys() as keys:
        try:
            while True:
                k = keys.get(0.05)
                now = time.time()
                if k == "q":
                    break
                elif k == "w":
                    vesc.set_current(MAX_TELEOP_CURRENT_A * 0.5)
                    last_drive = now
                elif k == "s":
                    vesc.set_brake_current(MAX_TELEOP_BRAKE_A * 0.5)
                    last_drive = now
                elif k == " ":
                    vesc.set_brake_current(MAX_TELEOP_BRAKE_A)
                    last_drive = now
                elif k == "a":
                    steer = max(steer - 0.15, -1.0)
                elif k == "d":
                    steer = min(steer + 0.15, 1.0)
                if now - last_drive > 0.4:           # dead-man: keys released
                    vesc.set_current(0.0)
                steer *= 0.92                         # auto-recenter
                if steer >= 0.0:
                    vesc.set_servo(center + steer * (right - center))
                else:
                    vesc.set_servo(center + steer * (center - left))
        finally:
            vesc.set_current(0.0)
            vesc.set_servo(center)
    print("teleop ended, motor stopped.")


# --------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Step 6: VESC bench tools.")
    ap.add_argument("--port", type=str, default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--config", type=str, default="car_link.yaml")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    cfg: dict = {}
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        print(f"loaded {cfg_path}")
    cfg.setdefault("port", args.port)
    cfg.setdefault("baud", args.baud)

    try:
        vesc = VescUART(args.port, args.baud)
    except Exception as exc:
        print(f"ERROR: cannot open {args.port}: {exc}\n"
              "Is the ESC powered (battery, not just USB), is the cable data-"
              "capable, and is your user in the dialout group?\n"
              "  sudo usermod -aG dialout $USER   (then log out and back in)")
        return 2

    fw = vesc.get_fw_version()
    if fw is None:
        print("ERROR: no answer from the VESC. Check that App to Use includes "
              "UART in VESC Tool, and the baud rate matches (115200).")
        return 2
    print(f"connected: VESC firmware {fw} on {args.port}")
    v = vesc.get_values()
    if v is not None:
        print(f"battery {v.v_in:.1f} V"
              + (f"  << BELOW the 75100's 14 V minimum -- it will cut out! "
                 "Use a 4S+ pack." if v.v_in < MIN_VOLTAGE_WARN else ""))

    menu = ("\n[1] telemetry  [2] spin test  [3] steering trim  "
            "[4] drive ratio  [5] teleop  [6] save config  [q] quit\n> ")
    with vesc:
        while True:
            choice = input(menu).strip().lower()
            if choice == "1":
                telemetry(vesc)
            elif choice == "2":
                spin_test(vesc)
            elif choice == "3":
                steering_trim(vesc, cfg)
            elif choice == "4":
                drive_ratio(cfg)
            elif choice == "5":
                teleop(vesc, cfg)
            elif choice == "6":
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                print(f"wrote {cfg_path}: {cfg}")
            elif choice == "q":
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
