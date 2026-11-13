# Drone Swarm Autonomous

A two-drone disaster-response simulation for an earthquake and landslide scene. The Scout surveys the area and detects people from its live camera feed. The Ambulance receives validated incident locations, approaches each assigned site, performs a clearance-checked descent, and returns to its launch area. The project runs on Ubuntu with PX4 SITL, Gazebo Harmonic, MAVSDK-Python, and YOLO, without ROS.

## How the demo works

1. **Scout survey.** The Scout flies a predefined five-waypoint route, captures downward-facing RGB images, and returns to launch. YOLO detects people in the images. Visiting a waypoint alone does not confirm a detection.
2. **Localization and dispatch.** The system projects image detections to ground locations using camera calibration and time-aligned Scout pose and GPS telemetry. The coordinator retains observations with unverified locations and queues validated incidents for dispatch. Incident and mission messages use the project's MANET messaging layer with acknowledgements.
3. **Ambulance response.** The Ambulance can start after its first eligible assignment while the Scout is still surveying. It uses LiDAR during approach and checks downward clearance during descent. It can service subsequent eligible assignments in the same sortie before returning to launch. The demo descent is an approach maneuver near the incident, with landing at the end of the sortie.
4. **Monitoring.** The coordinator logs mission acknowledgement, descent, completed incidents, return-to-launch (RTL), and landing. Gazebo shows the simulation. Terminal logs provide the mission evidence.

The five simulated victim models provide offline evaluation ground truth. The detector and coordinator use live observations for incident locations. Fire-Response and the earlier MAPPO/RELAY experiments are outside the current two-drone demo.

## Demonstrated results

| Run | Outcome |
| --- | --- |
| Four-incident run, 2026-11-12 at 17:23–17:27 UTC | Four descents confirmed. One used an offset approach. The Ambulance returned after waiting for another eligible assignment and confirmed landing. |
| Run `run-20261113T052739Z-czdpMQ` | Three descents confirmed, including an offset approach. The Ambulance returned after an airborne handoff timeout, accepted RTL, and confirmed landing. |

**Five-victim completion remains unverified.** A run can finish early if the coordinator cannot validate enough locations, a clearance or telemetry check holds the descent, or another eligible assignment does not arrive before the handoff deadline. GUI and recording runs on the original laptop coincided with clock-mapping and pose-matching failures. The logs establish the localization failures, while the specific resource bottleneck remains unverified.

## Setup on Ubuntu

Clone the project:

```bash
git clone https://github.com/ganeshpishey/Drone-Swarm-Autonomous.git
cd Drone-Swarm-Autonomous
```

Install Gazebo Harmonic and build PX4 SITL separately. The launcher defaults to `/tmp/PX4-Autopilot`. If you built PX4 in another directory, export its path in the launch terminal:

```bash
export PX4_AUTOPILOT_DIR="$HOME/PX4-Autopilot"
```

Create the MAVSDK environment with Python 3.10. The launcher version used for this project looks for the MAVSDK server under the Python 3.10 package directory.

```bash
python3.10 -m venv .venv-mavsdk
.venv-mavsdk/bin/python -m pip install 'mavsdk-grpc>=3.17,<4'
```

The live detector uses `.venv-perception`, Ultralytics, and `yolo26n.pt` at the repository root. Install the detector's Gazebo Python transport bindings for the local Harmonic installation. The launcher also references `.venv-px4` for battery configuration. Review the launcher for its environment and asset checks when setting up a fresh machine.

## Run the two-drone demo

Run Terminal 1 from the repository root. Check the supported launch modes in your checkout with `./integration/mavsdk_controller/launch_active_uavs.sh --help`. The five-waypoint workflow uses:

```bash
./integration/mavsdk_controller/launch_active_uavs.sh final-five
```

In **Terminal 2**, monitor the Scout and Ambulance events:

```bash
RUN="$(readlink -e "${TMPDIR:-/tmp}/drone-swarm-active-uavs-$(id -u)/current")" &&
tail -n +1 -F "$RUN/logs/scout-survey.log" "$RUN/logs/coordinator.log" |
grep --line-buffered -E 'SCOUT_WAYPOINT_COMPLETE|SCOUT_SURVEY_COMPLETED_AND_LANDED|MISSION APPLICATION ACK|AMBULANCE (FLIGHT RUNNER|DESCENT|VICTIM COMPLETED|MULTI VICTIM RETURN|RTL ACCEPTED|LANDED CONFIRMED|RECOVERY REQUIRED)'
```

`MISSION APPLICATION ACK` with `accepted_standby` confirms receipt of an assignment. Look for `AMBULANCE DESCENT CONFIRMED` and `AMBULANCE VICTIM COMPLETED` for each serviced incident, and `AMBULANCE LANDED CONFIRMED` for the end of the sortie.

In **Terminal 3**, attach Gazebo to the active run:

```bash
RUN="$(readlink -e "${TMPDIR:-/tmp}/drone-swarm-active-uavs-$(id -u)/current")" &&
source "$RUN/gz_env.sh" &&
gz sim -g
```

From the repository root, stop the session after landing with `./integration/mavsdk_controller/launch_active_uavs.sh stop`. Save any logs or recordings you want to keep from the temporary run directory before cleanup.

## Repository map

| Path | Purpose |
| --- | --- |
| `worlds/earthquake_landslide_apron_world.sdf` | Disaster-response world used by the two-drone launcher |
| `worlds/earthquake_landslide_apron_ground_truth.json` | Five victim model poses for offline evaluation |
| `worlds/disaster_assets/` | Terrain, scene assets, and drone models |
| `integration/mavsdk_controller/launch_active_uavs.sh` | Scout and Ambulance SITL launcher |
| `integration/mavsdk_controller/scout_victim_survey.py` | Scout waypoint survey and return |
| `integration/mavsdk_controller/scout_live_person_detection.py` | Live RGB inference |
| `integration/mavsdk_controller/scout_camera_gps_localization.py` | Camera and GPS ground projection |
| `integration/mavsdk_controller/day10_coordinator_service.py` | Incident coordination, dispatch, and monitoring |
| `integration/mavsdk_controller/ambulance_flight_runner.py` | Ambulance flight execution |
| `docs/DRONE_SWARM_14_DAY_PLAN.md` | Development roadmap and earlier acceptance criteria |
| `docs/RELAY_LEGACY_README.md` | Setup for the separate MAPPO/RELAY experiments |


