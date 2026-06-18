"""Minimal VESC UART driver (firmware 5.x / 6.x), tested against fw 6.02.

Implements just what the car needs -- no external VESC libraries:

    COMM_FW_VERSION      handshake / connection check
    COMM_GET_VALUES      telemetry (voltage, ERPM, currents, temps, odometer)
    COMM_SET_DUTY        gentle bench testing
    COMM_SET_CURRENT     drive torque command
    COMM_SET_CURRENT_BRAKE  braking
    COMM_SET_RPM         closed-loop ERPM (speed) command
    COMM_SET_SERVO_POS   steering servo on the ESC's PPM pin (servo-out mode)
    COMM_ALIVE           watchdog heartbeat

Framing: 0x02 | len (1 byte) | payload | crc16-xmodem(payload) | 0x03
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass

# COMM packet ids (stable low ids of the VESC firmware)
COMM_FW_VERSION = 0
COMM_GET_VALUES = 4
COMM_SET_DUTY = 5
COMM_SET_CURRENT = 6
COMM_SET_CURRENT_BRAKE = 7
COMM_SET_RPM = 8
COMM_SET_SERVO_POS = 12
COMM_ALIVE = 30


def crc16(data: bytes) -> int:
    """CRC16/XMODEM (poly 0x1021, init 0), as used by the VESC firmware."""
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
    return crc


def encode_frame(payload: bytes) -> bytes:
    if len(payload) > 255:
        raise ValueError("long packets not supported")
    c = crc16(payload)
    return bytes([0x02, len(payload)]) + payload + bytes([c >> 8, c & 0xFF, 0x03])


@dataclass
class VescValues:
    temp_fet_c: float
    temp_motor_c: float
    motor_current_a: float
    input_current_a: float
    duty: float
    erpm: float
    v_in: float
    tachometer: int          # counts; /(3 * pole_pairs) = motor revolutions
    tachometer_abs: int
    fault: int


def parse_values(payload: bytes) -> VescValues:
    """Payload starts with the COMM_GET_VALUES id byte."""
    u = struct.unpack_from
    return VescValues(
        temp_fet_c=u(">h", payload, 1)[0] / 10.0,
        temp_motor_c=u(">h", payload, 3)[0] / 10.0,
        motor_current_a=u(">i", payload, 5)[0] / 100.0,
        input_current_a=u(">i", payload, 9)[0] / 100.0,
        # avg_id (13) and avg_iq (17) skipped
        duty=u(">h", payload, 21)[0] / 1000.0,
        erpm=float(u(">i", payload, 23)[0]),
        v_in=u(">h", payload, 27)[0] / 10.0,
        # amp/watt hour counters (29..44) skipped
        tachometer=u(">i", payload, 45)[0],
        tachometer_abs=u(">i", payload, 49)[0],
        fault=payload[53],
    )


class VescUART:
    """Blocking serial link to a VESC.  Use as a context manager."""

    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 0.3):
        import serial  # lazy: only the car needs pyserial
        self.ser = serial.Serial(port, baud, timeout=timeout_s)

    # ------------------------------------------------------------ plumbing

    def close(self) -> None:
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.set_current(0.0)
        finally:
            self.close()

    def _send(self, payload: bytes) -> None:
        self.ser.write(encode_frame(payload))

    def _recv(self, expected_id: int, deadline_s: float = 0.5) -> bytes | None:
        """Scan the stream for a valid frame with the expected id."""
        end = time.time() + deadline_s
        buf = b""
        while time.time() < end:
            buf += self.ser.read(64)
            start = buf.find(b"\x02")
            if start < 0 or len(buf) < start + 2:
                continue
            length = buf[start + 1]
            frame_end = start + 2 + length + 3
            if len(buf) < frame_end:
                continue
            payload = buf[start + 2: start + 2 + length]
            crc_rx = (buf[start + 2 + length] << 8) | buf[start + 3 + length]
            if (buf[frame_end - 1] == 0x03 and crc_rx == crc16(payload)
                    and payload and payload[0] == expected_id):
                return payload
            buf = buf[start + 1:]          # resync past the bad start byte
        return None

    # ------------------------------------------------------------ commands

    def get_fw_version(self) -> str | None:
        self._send(bytes([COMM_FW_VERSION]))
        payload = self._recv(COMM_FW_VERSION)
        if payload is None or len(payload) < 3:
            return None
        return f"{payload[1]}.{payload[2]}"

    def get_values(self) -> VescValues | None:
        self._send(bytes([COMM_GET_VALUES]))
        payload = self._recv(COMM_GET_VALUES)
        return parse_values(payload) if payload and len(payload) >= 54 else None

    def set_duty(self, duty: float) -> None:
        self._send(struct.pack(">Bi", COMM_SET_DUTY, int(duty * 100000)))

    def set_current(self, amps: float) -> None:
        self._send(struct.pack(">Bi", COMM_SET_CURRENT, int(amps * 1000)))

    def set_brake_current(self, amps: float) -> None:
        self._send(struct.pack(">Bi", COMM_SET_CURRENT_BRAKE, int(amps * 1000)))

    def set_erpm(self, erpm: float) -> None:
        self._send(struct.pack(">Bi", COMM_SET_RPM, int(erpm)))

    def set_servo(self, position: float) -> None:
        """Steering servo position 0..1 (requires servo-out enabled)."""
        position = min(max(position, 0.0), 1.0)
        self._send(struct.pack(">BH", COMM_SET_SERVO_POS, int(position * 1000)))

    def alive(self) -> None:
        self._send(bytes([COMM_ALIVE]))
