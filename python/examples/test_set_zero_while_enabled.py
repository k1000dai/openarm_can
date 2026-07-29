#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Does a Damiao motor accept the "set zero" command (0xFE) while it is ENABLED
# and producing torque?
#
# openarm-can-zero-position-calibration writes zeros only after disable_all().
# openarm-can-cli set-zero does Disable -> 0xFE -> Disable. Neither tells us
# whether 0xFE is honored on a live, torque-producing motor, which is what a
# per-joint "zero at the mechanical stop" calibration would need.
#
# Only J7 is initialized here, so no other joint is ever enabled.
#
# WARNING: this overwrites J7's stored zero position. Re-run the full
# calibration afterwards.

import argparse
import sys
import time

import numpy as np
import openarm_can as oa

# J7 defaults. Note the hit thresholds below are the ones the calibration
# script uses for non-J1 arm joints; J1 (DM8009) needs different ones.
DEFAULT_SEND_ID = 0x07
DEFAULT_RECV_ID = 0x17
HIT_DQ_TH = 0.1   # rad/s
HIT_TAU_TH = 2.0  # Nm

HOLD_KP, HOLD_KD = 30.0, 0.8
BUMP_KP, BUMP_KD = 45.0, 1.2
SAFE_KD = 0.8  # damping-only gain used the instant after 0xFE

# A verdict needs the pre-write position to be unambiguously far from zero.
SEPARATION_RAD = 0.15


def read_state(openarm, motor):
    """Poll the motor and return (q, dq, tau)."""
    openarm.refresh_all()
    openarm.recv_all()
    return motor.get_position(), motor.get_velocity(), motor.get_torque()


def hold(openarm, arm, q, kp, kd, duration=0.3):
    """Keep sending the same MIT setpoint so the motor stays put."""
    n = max(1, int(duration / 0.005))
    for _ in range(n):
        arm.mit_control_one(0, oa.MITParam(kp, kd, q, 0.0, 0.0))
        openarm.recv_all()
        time.sleep(0.005)


def relax(openarm, arm, duration=0.3):
    """Damping only: no position reference, so no jump whatever the frame is."""
    n = max(1, int(duration / 0.005))
    for _ in range(n):
        arm.mit_control_one(0, oa.MITParam(0.0, SAFE_KD, 0.0, 0.0, 0.0))
        openarm.recv_all()
        time.sleep(0.005)


def move_to(openarm, arm, motor, q_target, kp, kd, interp_time=2.0):
    """Linear interpolation from the current position to q_target."""
    openarm.recv_all()
    q0 = motor.get_position()
    n_steps = 400
    dt = interp_time / n_steps
    for i in range(n_steps + 1):
        q = q0 + (q_target - q0) * (i / n_steps)
        arm.mit_control_one(0, oa.MITParam(kp, kd, q, 0.0, 0.0))
        openarm.recv_all()
        time.sleep(dt)
    hold(openarm, arm, q_target, kp, kd, duration=0.5)


def bump_to_limit(openarm, arm, motor, step_deg, max_travel_deg, kp, kd):
    """Step until the mechanical stop. Returns (hit, q_at_contact, tau_at_contact).

    Unlike the calibration script's version this one gives up instead of
    pushing forever.
    """
    step_rad = np.deg2rad(step_deg)
    max_travel_rad = abs(np.deg2rad(max_travel_deg))
    openarm.recv_all()
    q_start = motor.get_position()
    q_target = q_start

    while abs(q_target - q_start) < max_travel_rad:
        q_target += step_rad
        arm.mit_control_one(0, oa.MITParam(kp, kd, q_target, 0.0, 0.0))
        openarm.recv_all()
        time.sleep(0.005)

        q, dq, tau = motor.get_position(), motor.get_velocity(), motor.get_torque()
        if abs(dq) < HIT_DQ_TH and abs(tau) > HIT_TAU_TH:
            print(f"  contact at q={q:+.4f} rad ({np.rad2deg(q):+.2f} deg), "
                  f"tau={tau:+.3f} Nm")
            return True, q, tau

    print(f"  [WARN] no contact within {max_travel_deg:.1f} deg of travel")
    return False, motor.get_position(), motor.get_torque()


def make_raw_socket(canport):
    """Second socket for the CLI-style classic-CAN 0xFE frame.

    zero_position_commands.cpp opens its own CANSocket(interface, false) for
    exactly this, so classic CAN is the proven path for config frames.
    """
    return oa.CANSocket(canport, False)


def fire_set_zero(openarm, arm, motor, method, raw_socket):
    """Send 0xFE while the motor is live, then immediately drop to damping only.

    Returns the position reported after the write.
    """
    if method == "lib":
        # Only J7 was initialized, so this reaches J7 and nothing else.
        openarm.set_zero_all()
    else:
        frame = oa.CanFrame()
        frame.can_id = motor.get_send_can_id()
        frame.data = bytes([0xFF] * 7 + [0xFE])
        raw_socket.write_can_frame(frame)

    # Re-base before anything can act on a stale setpoint. kp=0 means there is
    # no position error term at all, so this is safe whether or not the write
    # landed.
    relax(openarm, arm, duration=0.4)
    q, _, _ = read_state(openarm, motor)
    return q


def verdict(label, q_before, q_after):
    """Decide whether the write landed, from the reported position alone."""
    print(f"\n  --- {label} ---")
    print(
        f"  position before 0xFE : {q_before:+.4f} rad ({np.rad2deg(q_before):+.2f} deg)")
    print(
        f"  position after  0xFE : {q_after:+.4f} rad ({np.rad2deg(q_after):+.2f} deg)")

    if abs(q_before) < SEPARATION_RAD:
        print("  => INCONCLUSIVE: pre-write position was too close to zero.")
        return "inconclusive"
    if abs(q_after) < SEPARATION_RAD:
        print("  => ACCEPTED: the motor now reports ~0, so 0xFE took effect "
              "while enabled.")
        return "accepted"
    if abs(q_after - q_before) < SEPARATION_RAD:
        print("  => IGNORED: the frame did not shift, so 0xFE was dropped "
              "while enabled.")
        return "ignored"
    print("  => UNEXPECTED: the position moved but not to zero. The joint "
          "probably drifted; re-run and inspect.")
    return "unexpected"


def phase_hold(openarm, arm, motor, args, raw_socket):
    """0xFE while enabled and holding a position (little or no load torque)."""
    print("\n[PHASE 1] enabled + holding a position")
    q0, _, _ = read_state(openarm, motor)
    print(f"  start position: {q0:+.4f} rad ({np.rad2deg(q0):+.2f} deg)")

    offset = np.deg2rad(args.offset_deg)
    q_target = q0 + offset
    if abs(q_target) < SEPARATION_RAD:
        # Would land too near zero to tell "accepted" from "ignored".
        q_target = q0 - offset
        print("  (target flipped so the pre-write position stays clear of 0)")
    print(f"  moving to {q_target:+.4f} rad ({np.rad2deg(q_target):+.2f} deg)")
    move_to(openarm, arm, motor, q_target, HOLD_KP, HOLD_KD)

    q_before, _, tau_before = read_state(openarm, motor)
    print(f"  holding at {q_before:+.4f} rad, tau={tau_before:+.3f} Nm, "
          f"enabled={motor.is_enabled()}")

    q_after = fire_set_zero(openarm, arm, motor, args.method, raw_socket)
    print(f"  enabled after write: {motor.is_enabled()}")
    return verdict("PHASE 1 (holding)", q_before, q_after)


def phase_stop(openarm, arm, motor, args, raw_socket):
    """0xFE while pressed into the mechanical stop: the real calibration case."""
    print("\n[PHASE 2] enabled + pressed into the mechanical stop")
    hit, q_before, tau_before = bump_to_limit(
        openarm, arm, motor,
        step_deg=args.bump_step_deg,
        max_travel_deg=args.bump_max_deg,
        kp=BUMP_KP, kd=BUMP_KD)

    if not hit:
        print("  => SKIPPED: never reached a stop, so there is nothing to "
              "press against.")
        relax(openarm, arm, duration=0.5)
        return "skipped"

    print(
        f"  pressing with tau={tau_before:+.3f} Nm, enabled={motor.is_enabled()}")
    q_after = fire_set_zero(openarm, arm, motor, args.method, raw_socket)
    print(f"  enabled after write: {motor.is_enabled()}")
    return verdict("PHASE 2 (pressed into stop)", q_before, q_after)


def phase_still_controllable(openarm, arm, motor, away_sign):
    """After the write, does the motor still track MIT commands normally?

    away_sign points away from whichever stop phase 2 pressed into, so this
    never drives the joint further into it.
    """
    print("\n[PHASE 3] is the motor still controllable after the write?")
    q_before, _, _ = read_state(openarm, motor)
    q_target = q_before + away_sign * np.deg2rad(5.0)
    move_to(openarm, arm, motor, q_target, HOLD_KP, HOLD_KD, interp_time=1.5)
    q_after, _, _ = read_state(openarm, motor)
    err = abs(q_after - q_target)
    print(f"  commanded {q_target:+.4f}, reached {q_after:+.4f}, "
          f"error {err:.4f} rad ({np.rad2deg(err):.2f} deg)")
    if err < np.deg2rad(3.0):
        print("  => OK: still tracking commands.")
        return "ok"
    print("  => DEGRADED: the motor did not follow. Check for a fault state.")
    return "degraded"


def main():
    parser = argparse.ArgumentParser(
        description="Test whether 0xFE (set zero) is honored on an ENABLED "
                    "motor. J7 only.")
    parser.add_argument('--canport', type=str, default='can0')
    parser.add_argument('--send-id', type=lambda s: int(s, 0),
                        default=DEFAULT_SEND_ID)
    parser.add_argument('--recv-id', type=lambda s: int(s, 0),
                        default=DEFAULT_RECV_ID)
    parser.add_argument('--method', type=str, default='lib',
                        choices=['lib', 'raw'],
                        help="lib: OpenArm.set_zero_all() over the normal "
                             "transport. raw: CLI-style classic-CAN frame on a "
                             "second socket.")
    parser.add_argument('--offset-deg', type=float, default=20.0,
                        help="how far to move before the phase 1 write")
    parser.add_argument('--bump-step-deg', type=float, default=0.2,
                        help="phase 2 step size; negative to bump the other way")
    parser.add_argument('--bump-max-deg', type=float, default=120.0,
                        help="phase 2 gives up after this much travel")
    parser.add_argument('--skip-stop', action='store_true',
                        help="skip phase 2 (no bump into the mechanical stop)")
    parser.add_argument('--yes', action='store_true',
                        help="required: acknowledges that J7's zero is overwritten")
    args = parser.parse_args()

    print("=" * 70)
    print("  0xFE-while-enabled test  (J7 only)")
    print("=" * 70)
    print(f"  canport   : {args.canport}")
    print(f"  motor ids : send=0x{args.send_id:02X} recv=0x{args.recv_id:02X}")
    print(f"  method    : {args.method}")
    print()
    print("  THIS OVERWRITES J7's STORED ZERO POSITION.")
    print("  Re-run openarm-can-zero-position-calibration afterwards.")
    print()
    print("  Only J7 is initialized, so the other six joints are never")
    print("  enabled and stay limp. Make sure the arm is resting somewhere")
    print("  it cannot fall from, and that J7 is free to rotate.")
    print("=" * 70)

    if not args.yes:
        print("\nRefusing to run without --yes.")
        return 1

    openarm = oa.OpenArm(args.canport, True)
    openarm.init_arm_motors([oa.MotorType.DM4310],
                            [args.send_id],
                            [args.recv_id])
    openarm.set_callback_mode_all(oa.CallbackMode.STATE)

    raw_socket = make_raw_socket(
        args.canport) if args.method == 'raw' else None

    results = {}
    arm = None
    try:
        print("\nEnabling J7...")
        openarm.enable_all()
        openarm.recv_all()
        time.sleep(0.1)

        arm = openarm.get_arm()
        motor = arm.get_motors()[0]
        q, _, _ = read_state(openarm, motor)
        print(f"  enabled={motor.is_enabled()}, q={q:+.4f} rad")
        if not motor.is_enabled():
            print("  [ERROR] J7 did not enable. Check wiring and CAN IDs.")
            return 1

        hold(openarm, arm, q, HOLD_KP, HOLD_KD, duration=0.5)

        results['phase1'] = phase_hold(openarm, arm, motor, args, raw_socket)
        away_sign = 1.0
        if not args.skip_stop:
            results['phase2'] = phase_stop(
                openarm, arm, motor, args, raw_socket)
            away_sign = -np.sign(args.bump_step_deg)
        results['phase3'] = phase_still_controllable(
            openarm, arm, motor, away_sign)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C -> stopping")
    finally:
        if arm is not None:
            relax(openarm, arm, duration=0.2)
        openarm.disable_all()
        openarm.recv_all()
        print("\n[INFO] J7 disabled.")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    for k in ('phase1', 'phase2', 'phase3'):
        if k in results:
            print(f"    {k}: {results[k]}")
    print()
    if results.get('phase2') == 'accepted' or (
            args.skip_stop and results.get('phase1') == 'accepted'):
        print("  0xFE IS honored on an enabled motor -> per-joint zeroing at")
        print("  the mechanical stop is viable. The command setpoint must")
        print("  still be re-based in the same breath as the write.")
    elif 'ignored' in results.values():
        print("  0xFE is dropped while enabled -> per-joint zeroing needs a")
        print("  disable/enable window, or the closed-loop approach instead.")
    else:
        print("  Result not clear-cut. See the per-phase output above.")
    print()
    print("  REMINDER: J7's zero is now invalid. Re-run the calibration.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
