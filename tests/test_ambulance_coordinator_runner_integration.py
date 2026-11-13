from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from integration.mavsdk_controller.day10_coordinator_service import Day10CoordinatorService


class _FakeRunner:
    captured = None

    def __init__(self, **kwargs):
        type(self).captured = kwargs
        self.state = "ARRIVAL_CONFIRMED"
        self.flight_commands_sent = False

    def arrival_confirmed(self):
        return False

    async def run(self):
        # Exercise the coordinator-provided guidance stream without touching
        # a MAVSDK object or issuing flight commands.
        stream_factory = type(self).captured["guidance_stream"]
        stream = stream_factory()
        guidance = await anext(stream)
        type(self).captured["test_guidance"] = guidance
        await stream.aclose()
        type(self).captured["test_arrival_callback"] = type(self).captured["arrival_confirmed"]()
        type(self).captured["test_drone"] = type(self).captured["drone"]
        type(self).stop_event.set()
        return {
            "state": "ARRIVAL_CONFIRMED",
            "arrival_confirmed": True,
            "flight_commands_sent": False,
            "rtl_requested": False,
        }


class AmbulanceCoordinatorRunnerIntegrationTests(unittest.TestCase):
    def test_acknowledged_active_mission_is_wired_to_runner_and_arrival_stays_with_coordinator(self):
        async def scenario():
            mission_id = "ambulance-mission-stage3"
            guidance = SimpleNamespace(
                mission_id=mission_id, state="CLEAR", reason="test_guidance",
                north_m_s=0.2, east_m_s=0.0, down_m_s=0.0,
            )
            mission = SimpleNamespace(
                mission_id=mission_id,
                vehicle_id="ambulance_0",
                state="acknowledged",
                target={"coordinates": [1.0, 2.0, -0.2]},
            )
            stop_event = asyncio.Event()
            drone = object()
            return_lifecycle = SimpleNamespace(state="IDLE")
            adapter = SimpleNamespace(setpoint_ownership=object())
            service = Day10CoordinatorService.__new__(Day10CoordinatorService)
            service.args = SimpleNamespace()
            service.stop_event = stop_event
            service.coordinator = SimpleNamespace(missions={mission_id: mission})
            service.active_ambulance_mission_id = mission_id
            service.mission_ack_times = {mission_id: 1.0}
            service.ambulance_sortie_state = "ACTIVE"
            service.vehicle_state = {"ambulance_0": {}}
            service.telemetry_times = {"ambulance_0": {}}
            service.nav_times = {"ambulance_0:position_velocity_ned": time.monotonic()}
            service.ambulance_nav = {
                "north_m": 1.0, "east_m": 2.0, "down_m": -3.0,
            }
            service.node = SimpleNamespace(liveness=SimpleNamespace(available=lambda *a, **k: True))
            service.scan_feed = None
            service.ambulance_adapter = adapter
            service.ambulance_return_lifecycle = return_lifecycle
            service.ambulance_guidance = guidance
            service._ambulance_guidance_revision = 1
            service.pending_incident_ids = []
            service._ambulance_runner = None
            service._event = lambda *a, **k: None
            service._persist_dispatch_queue = lambda: None
            service._write_state = lambda: None
            service._battery_status = lambda *a, **k: {"level": "UNKNOWN"}
            _FakeRunner.captured = None
            _FakeRunner.stop_event = stop_event

            with patch(
                "integration.mavsdk_controller.ambulance_flight_runner.AmbulanceFlightRunner",
                _FakeRunner,
            ), patch(
                "integration.mavsdk_controller.ambulance_offboard_session.AmbulanceOffboardSession",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
            ), patch(
                "integration.mavsdk_controller.ambulance_sortie_executor.AmbulanceSortieExecutor",
                side_effect=lambda config: SimpleNamespace(config=config),
            ), patch(
                "integration.mavsdk_controller.ambulance_sortie_executor.ExecutorConfig",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
            ), patch(
                "integration.mavsdk_controller.ambulance_flight_runner.FlightRunnerConfig",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
            ):
                await service._monitor_ambulance_flight_execution(drone)

            captured = _FakeRunner.captured
            self.assertIsNotNone(captured)
            self.assertIs(captured["drone"], drone)
            self.assertIs(captured["return_lifecycle"], return_lifecycle)
            self.assertIs(captured["test_guidance"], guidance)
            self.assertTrue(captured["test_arrival_callback"])
            self.assertIs(captured["test_drone"], drone)
            self.assertEqual(service.ambulance_sortie_state, "ARRIVAL_CONFIRMED")
            self.assertIsNone(service._ambulance_runner)
            self.assertFalse(captured["test_guidance"] is None)

        asyncio.run(scenario())


    def test_five_confirmed_descents_request_one_final_return(self):
        async def scenario():
            service = Day10CoordinatorService.__new__(Day10CoordinatorService)
            service.args = SimpleNamespace(
                sitl_descent_demo=True, sitl_approach_demo=True,
                reserve_battery_percent=50,
            )
            service.stop_event = asyncio.Event()
            service.ambulance_return_lifecycle = SimpleNamespace(state="IDLE")
            service.ambulance_adapter = SimpleNamespace(setpoint_ownership=object())
            service.ambulance_sortie_state = "ACTIVE"
            service._concurrent_descent_enabled = lambda: True
            service._scout_clear_for_ambulance = lambda: True
            service.vehicle_state = {
                "scout_0": {"in_air": True},
                "ambulance_0": {
                    "position": {"relative_altitude_m": 3.3},
                    "attitude_euler": {"roll_deg": 0, "pitch_deg": 0},
                },
            }
            service.scan_feed = SimpleNamespace()
            service.ambulance_guidance = SimpleNamespace(state="GOAL")
            service.ambulance_nav = {
                "north_m": 0.0, "east_m": 0.0, "down_m": -2.55,
                "north_velocity_m_s": 0.0, "east_velocity_m_s": 0.0,
            }
            service.nav_times = {}
            service._persist_dispatch_queue = lambda: None
            service._write_state = lambda: None
            service._battery_status = lambda _: {"fresh": True, "remaining_percent": 95}
            service.mission_ack_times = {}
            events = []
            continuation_flags = []

            def event(name, **details):
                events.append((name, details))
                if name == "ambulance_multi_victim_return":
                    service.stop_event.set()

            service._event = event
            missions = {
                str(i): SimpleNamespace(
                    mission_id=str(i), incident_id=f"victim-{i}", state="acknowledged",
                    target={"coordinates": [0.0, 0.0, 0.0]},
                ) for i in range(1, 6)
            }

            def transition(mission_id, state, **_):
                missions[mission_id].state = state
                return {"accepted": True}

            service.coordinator = SimpleNamespace(
                missions=missions, update_mission=transition,
                incidents={m.incident_id: {} for m in missions.values()},
            )
            service.active_ambulance_mission_id = "1"
            service.mission_ack_times["1"] = time.monotonic()
            service.pending_incident_ids = ["2", "3", "4", "5"]

            async def dispatch(*, airborne_continuation, completed_mission_id):
                self.assertTrue(airborne_continuation)
                self.assertEqual(missions[completed_mission_id].state, "completed")
                self.assertEqual(service.ambulance_sortie_state, "AIRBORNE_HANDOFF")
                mission_id = service.pending_incident_ids.pop(0)
                service.active_ambulance_mission_id = mission_id
                service.mission_ack_times[mission_id] = time.monotonic()
                service.ambulance_sortie_state = "ACTIVE"

            service._dispatch_next_ambulance_incident = dispatch

            class Scan:
                error = None
                latest = {"ranges": [2.5] * 81}

                @property
                def received_at(self):
                    return time.monotonic() - 0.01

                def __init__(self, *_, **__):
                    pass

                async def start(self):
                    pass

                async def stop(self):
                    pass

            class Runner:
                def __init__(self, **kwargs):
                    self.guidance = kwargs["guidance_stream"]
                    self.state = "ARRIVAL_CONFIRMED"

                def arrival_confirmed(self):
                    return True

                async def run(self, *, airborne_continuation=False):
                    continuation_flags.append(airborne_continuation)
                    service.ambulance_nav["down_m"] = -2.55
                    service.scan_feed.received_at = time.monotonic()
                    service.nav_times["ambulance_0:position_velocity_ned"] = time.monotonic()
                    async for directive in self.guidance():
                        if directive.state == "CLIMB":
                            service.ambulance_nav["down_m"] = -3.3
                        if directive.state == "DESCEND":
                            service.ambulance_nav["down_m"] = -2.23
                    return {"arrival_confirmed": True}

            with patch(
                "integration.mavsdk_controller.ambulance_flight_runner.AmbulanceFlightRunner", Runner,
            ), patch(
                "integration.mavsdk_controller.day10_coordinator_service.GazeboScanFeed", Scan,
            ):
                await asyncio.wait_for(service._monitor_ambulance_flight_execution(object()), timeout=8)
            self.assertEqual(continuation_flags, [False, True, True, True, True])
            self.assertEqual(sum(name == "ambulance_descent_confirmed" for name, _ in events), 5)
            returns = [details for name, details in events if name == "ambulance_multi_victim_return"]
            self.assertEqual([item["reason"] for item in returns], ["five_descents_confirmed"])
            self.assertEqual(returns[0]["confirmed_victims"], 5)
            self.assertEqual(service.ambulance_sortie_state, "RETURN_REQUIRED")
            self.assertEqual(missions["5"].state, "returning")

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
