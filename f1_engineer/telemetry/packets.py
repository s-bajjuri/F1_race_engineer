from __future__ import annotations

import struct

from f1_engineer.models import (
    PacketHeader,
    PlayerDamage,
    PlayerLapData,
    PlayerMotion,
    PlayerMotionEx,
    PlayerStatus,
    PlayerTelemetry,
)

F1_25_PACKET_FORMAT = 2025

MOTION_PACKET_ID = 0
LAP_DATA_PACKET_ID = 2
CAR_TELEMETRY_PACKET_ID = 6
CAR_STATUS_PACKET_ID = 7
CAR_DAMAGE_PACKET_ID = 10
MOTION_EX_PACKET_ID = 13

HEADER_STRUCT = struct.Struct("<HBBBBBQfIIBB")  # 29 bytes
LAP_DATA_RECORD_SIZE = 57
LAP_DATA_PREFIX_STRUCT = struct.Struct("<IIHBHBHBHBfffBBBBBB")
CAR_TELEMETRY_STRUCT = struct.Struct("<HfffBbHBBH4H4B4BH4f4B")  # 60 bytes
CAR_MOTION_STRUCT = struct.Struct("<6f6h6f")  # 60 bytes
CAR_STATUS_STRUCT = struct.Struct("<5B3f2H2BH3Bb3fB3fB")  # 55 bytes
CAR_DAMAGE_STRUCT = struct.Struct("<4f30B")  # 46 bytes
MOTION_EX_STRUCT = struct.Struct("<61f")  # Player car only, 244 bytes

REQUIRED_FRAME_PARTS = {
    "lap",
    "telemetry",
    "motion",
    "status",
    "motion_ex",
}

PACKET_NAMES = {
    MOTION_PACKET_ID: "Motion",
    LAP_DATA_PACKET_ID: "Lap Data",
    CAR_TELEMETRY_PACKET_ID: "Car Telemetry",
    CAR_STATUS_PACKET_ID: "Car Status",
    CAR_DAMAGE_PACKET_ID: "Car Damage",
    MOTION_EX_PACKET_ID: "Motion Ex",
}

HIGH_RATE_PACKET_IDS = (
    MOTION_PACKET_ID,
    LAP_DATA_PACKET_ID,
    CAR_TELEMETRY_PACKET_ID,
    CAR_STATUS_PACKET_ID,
    MOTION_EX_PACKET_ID,
)



def parse_header(packet: bytes) -> PacketHeader:
    if len(packet) < HEADER_STRUCT.size:
        raise ValueError("Packet is smaller than the F1 25 header.")

    values = HEADER_STRUCT.unpack_from(packet)
    return PacketHeader(
        packet_format=values[0],
        game_major_version=values[2],
        game_minor_version=values[3],
        packet_id=values[5],
        session_uid=values[6],
        session_time=values[7],
        frame_identifier=values[8],
        overall_frame_identifier=values[9],
        player_car_index=values[10],
    )

def _player_record_offset(header: PacketHeader, record_size: int) -> int:
    if not 0 <= header.player_car_index < 22:
        raise ValueError(f"Invalid player car index: {header.player_car_index}")
    return HEADER_STRUCT.size + header.player_car_index * record_size

def parse_player_lap_data(packet: bytes, header: PacketHeader) -> PlayerLapData:
    offset = _player_record_offset(header, LAP_DATA_RECORD_SIZE)
    if len(packet) < offset + LAP_DATA_RECORD_SIZE:
        raise ValueError("Lap Data packet is too small.")

    values = LAP_DATA_PREFIX_STRUCT.unpack_from(packet, offset)
    return PlayerLapData(
        last_lap_time_ms=values[0],
        current_lap_time_ms=values[1],
        lap_distance_m=values[10],
        total_distance_m=values[11],
        car_position=values[13],
        current_lap_num=values[14],
        pit_status=values[15],
        sector=values[17],
        current_lap_invalid=bool(values[18]),
    )

def parse_player_telemetry(packet: bytes, header: PacketHeader) -> PlayerTelemetry:
    offset = _player_record_offset(header, CAR_TELEMETRY_STRUCT.size)
    if len(packet) < offset + CAR_TELEMETRY_STRUCT.size:
        raise ValueError("Car Telemetry packet is too small.")

    values = CAR_TELEMETRY_STRUCT.unpack_from(packet, offset)
    return PlayerTelemetry(
        speed_kmh=values[0],
        throttle=max(0.0, min(1.0, values[1])),
        steering=max(-1.0, min(1.0, values[2])),
        brake=max(0.0, min(1.0, values[3])),
        clutch=values[4],
        gear=values[5],
        rpm=values[6],
        drs=values[7],
        brake_temperatures=tuple(values[10:14]),
        tyre_surface_temperatures=tuple(values[14:18]),
        tyre_inner_temperatures=tuple(values[18:22]),
        engine_temperature=values[22],
        tyre_pressures=tuple(values[23:27]),
    )

def parse_player_motion(packet: bytes, header: PacketHeader) -> PlayerMotion:
    offset = _player_record_offset(header, CAR_MOTION_STRUCT.size)
    if len(packet) < offset + CAR_MOTION_STRUCT.size:
        raise ValueError("Motion packet is too small.")

    # CarMotionData layout ("<6f6h6f"):
    #   0-2   worldPosition X/Y/Z      (Y is elevation)
    #   3-5   worldVelocity X/Y/Z
    #   6-11  worldForwardDir / worldRightDir (int16; orientation is already
    #         fully described by yaw/pitch/roll below, so they are not stored)
    #   12-14 gForce lateral / longitudinal / vertical
    #   15-17 yaw / pitch / roll       (radians)
    values = CAR_MOTION_STRUCT.unpack_from(packet, offset)
    return PlayerMotion(
        world_position_x=values[0],
        world_position_y=values[1],
        world_position_z=values[2],
        world_velocity_x=values[3],
        world_velocity_y=values[4],
        world_velocity_z=values[5],
        g_force_lateral=values[12],
        g_force_longitudinal=values[13],
        g_force_vertical=values[14],
        yaw=values[15],
        pitch=values[16],
        roll=values[17],
    )

def parse_player_status(packet: bytes, header: PacketHeader) -> PlayerStatus:
    offset = _player_record_offset(header, CAR_STATUS_STRUCT.size)
    if len(packet) < offset + CAR_STATUS_STRUCT.size:
        raise ValueError("Car Status packet is too small.")

    values = CAR_STATUS_STRUCT.unpack_from(packet, offset)
    return PlayerStatus(
        fuel_in_tank=values[5],
        ers_deployed_this_lap=values[23],
    )

def parse_player_damage(packet: bytes, header: PacketHeader) -> PlayerDamage:
    offset = _player_record_offset(header, CAR_DAMAGE_STRUCT.size)
    if len(packet) < offset + CAR_DAMAGE_STRUCT.size:
        raise ValueError("Car Damage packet is too small.")

    values = CAR_DAMAGE_STRUCT.unpack_from(packet, offset)
    tyre_wear_fraction = tuple(float(value) / 100.0 for value in values[:4])
    return PlayerDamage(tyre_wear=tyre_wear_fraction)

def parse_player_motion_ex(packet: bytes) -> PlayerMotionEx:
    offset = HEADER_STRUCT.size
    if len(packet) < offset + MOTION_EX_STRUCT.size:
        raise ValueError("Motion Ex packet is too small.")

    # PacketMotionExData is 61 floats for the player car. Group offsets:
    #   0-3 suspPosition   4-7 suspVelocity    8-11 suspAcceleration
    #   12-15 wheelSpeed   16-19 slipRatio     20-23 slipAngle
    #   24-27 latForce     28-31 longForce     32 heightOfCOGAboveGround
    #   33-35 localVelocity X/Y/Z              36-38 angularVelocity X/Y/Z
    #   39-41 angularAcceleration X/Y/Z        42 frontWheelsAngle
    # angularVelocityY is yaw rate: the measured rotation the slip angles and
    # steering input have to be judged against.
    values = MOTION_EX_STRUCT.unpack_from(packet, offset)
    return PlayerMotionEx(
        wheel_slip_ratio=tuple(values[16:20]),
        wheel_slip_angle=tuple(values[20:24]),
        local_velocity_x=values[33],
        local_velocity_y=values[34],
        local_velocity_z=values[35],
        yaw_rate=values[37],
        front_wheels_angle=values[42],
    )
