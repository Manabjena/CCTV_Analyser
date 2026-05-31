Markdown
# Architectural Design Document: Store Intelligence Platform

This document describes the end-to-end system architecture of the Store Intelligence Platform deployed for Apex Retail at store ST1008 (Brigade Road, Bangalore). The platform transforms raw surveillance feeds from a 5-camera topology into a clean stream of real-time relational retail events to compute accurate operational metrics, specifically the Store Conversion Rate.

## 1. End-to-End System Architecture

The system uses a decoupled architecture divided into two distinct processing boundaries: the Edge Inference Pipeline and the Centralized Intelligence API.

+-----------------------------------------------------------------------------------+
|                            EDGESurveillance Inference Loop                        |
|                                                                                   |
|  [CAM 1] [CAM 2] [CAM 3] [CAM 4] [CAM 5]                                          |
|       │       │       │       │       │                                           |
|       ▼       ▼       ▼       ▼       ▼                                           |
|  ┌─────────────────────────────────────────────────────────────────────────────┐  |
|  │ 1. MultiCamStreamMultiplexer (Frame Synchronisation & Sliding Windows)      │  |
|  └─────────────────────────────────────────────────────────────────────────────┘  |
|       │                                                                           |
|       ▼ [Synchronised 5-Frame Arrays]                                             |
|  ┌─────────────────────────────────────────────────────────────────────────────┐  |
|  │ 2. BatchedStoreDetector (YOLO Parallel Bounding Box Inference)              │  |
|  └─────────────────────────────────────────────────────────────────────────────┘  |
|       │                                                                           |
|       ▼ [Raw Unassociated Coordinates]                                            |
|  ┌─────────────────────────────────────────────────────────────────────────────┐  |
|  │ 3. GlobalMultiCamReIDTracker (Hungarian Matrix Association + OSNet/Hist Dim) │  |
|  └─────────────────────────────────────────────────────────────────────────────┘  |
|       │                                                                           |
|       ▼ [Unified Global Identity Trajectories]                                    |
|  ┌─────────────────────────────────────────────────────────────────────────────┐  |
|  │ 4. SpatialEventEmitter (Directional Boundary Crossings & Dwell Clocks)      │  |
|  └─────────────────────────────────────────────────────────────────────────────┘  |
+-----------------------------------------------------------------------------------+
│
▼ HTTP POST /events/ingest (Port 8000)
+-----------------------------------------------------------------------------------+
|                        CENTRALIZED INTELLIGENCE API SURFACE                       |
|                                                                                   |
|  ┌─────────────────────────────────────────────────────────────────────────────┐  |
|  │ 1. Serialization Gateway (Pydantic Enum Validation & Strict Type Contracts) │  |
|  └─────────────────────────────────────────────────────────────────────────────┘  |
|       │                                                                           |
|       ▼ [Validated Payloads]                                                      |
|  ┌─────────────────────────────────────────────────────────────────────────────┐  |
|  │ 2. ThreadSafeIngestionEngine (Mutex Lock-Protected Storage & Memory Sweeps) │  |
|  └─────────────────────────────────────────────────────────────────────────────┘  |
|       │                                                                           |
|       ▼ [Snapshot Extraction Pass]                                                |
|  ┌─────────────────────────────────────────────────────────────────────────────┐  |
|  │ 3. TransactionAnalyticsEngine (O(1) Temporally-Indexed Sliding POS Joins)   │  |
|  └─────────────────────────────────────────────────────────────────────────────┘  |
+-----------------------------------------------------------------------------------+


### 1.1 Edge Surveillance Inference Loop
* **Frame Synchronisation:** The `MultiCamStreamMultiplexer` opens parallel file descriptors for all 5 video streams simultaneously. It reads frames at a matched frame index to build a uniform timeline, matching frame 300 across all fields of view to preserve spatial-temporal integrity.
* **Parallel Inference:** Synchronized frames are aggregated into a single 4D tensor and processed in a parallel batch pass through the YOLO object detection model. This maximizes GPU thread utility and maintains low latency.
* **Identity Fusing:** Bounding boxes are passed to the `GlobalMultiCamReIDTracker`. This module uses localized Kalman Box Filters to handle single-camera trajectories, and runs a global Hungarian assignment optimization pass across visual similarity, temporal drift, and store layout constraints to assign a unique, long-term global ID to each customer.
* **State Machine Translation:** The `SpatialEventEmitter` maps the global tracking center points against the store's physical zone layout. It evaluates directional motion vectors to trigger transitions and tracks dwelling times at the frame level.

### 1.2 Centralized Intelligence API Surface
* **Data Validation:** The entry surface uses strict `Pydantic` validation contracts to ensure inbound JSON arrays conform to system requirements. Primitive strings are replaced with explicit system Enums, and timestamps are verified against strict ISO-8601 specifications.
* **Thread Safety:** Validated events are passed to the `ThreadSafeIngestionEngine`. State mutations are protected by an explicit mutual exclusion re-entrant lock (`threading.Lock`) to prevent data race conditions across multi-threaded web worker pools.
* **Idempotency & Garbage Collection:** Inbound request streams are screened against a fast $O(1)$ hash set ring buffer to instantly filter out duplicate network transmissions. A background retention policy sweep monitors frame indices, automatically evicting tracking histories that exceed a 450-frame limit to prevent system memory leaks.

---

## 2. Spatial Zoning & Camera Topology Layout

The deployment handles a 5-camera matrix monitoring store `ST1008` (Brigade Road, Bangalore). Each camera maps to a specific operational retail domain:

* **`CAM_3` (Gate Threshold):** Positioned outside the main entrance facing inward. It acts as the primary traffic monitor. Center-point velocity vectors crossing the horizontal mid-line ($y = 540$) trigger `ENTRY` or `EXIT` events to establish traffic baselines.
* **`CAM_1` (Main Floor Left):** Monitors the left side of the store layout. Bounding box coordinates are split vertically down the middle ($x = 960$) to track visits to the `MAKEUP` and `FRAGRANCE` departments.
* **`CAM_2` (Main Floor Right):** Monitors the right side of the storefront. Coordinates are split vertically down the center line ($x = 960$) to register interaction states for the `SKIN` and `HAIR` departments.
* **`CAM_5` (Billing Desk):** Dedicated exclusively to the checkout register queues. It tracks interactions within the checkout zone and computes real-time queue depths.
* **`CAM_4` (Storage Room):** Monitors the restricted stockroom and unauthorized spaces. It is used to separate customer behavior from operational workforce movements.

---

## 3. Core Edge-Case Mitigation Strategies

Real-world computer vision systems encounter frequent environmental anomalies. The tracking architecture uses several deterministic safeguards to maintain data accuracy:

* **Group Entries (Tripwire Crossing):** Customers entering close together can blend into a single visual cluster. The system addresses this by running high-frequency bounding-box centroid evaluations right at the entrance gate line. This ensures individual tracks are registered independently as they cross the threshold, preventing traffic deflation.
* **Staff Movement Isolation:** Employees wear uniforms that match the `staff_uniform` model classification. Once an identity is flagged as staff, the global tracker adds it to the persistent `staff_global_ids` registry. The system then filters out these IDs from consumer KPIs and queue depth metrics, while routing backroom actions to a separate `STAFF_BACKROOM_LOG` inside `CAM_4`.
* **Partial Occlusions (Lost State Recovery):** When a shopper is temporarily blocked by display racks or structural columns, single-camera trackers move the identity to a `Lost` state instead of dropping it immediately. The underlying Kalman Box Filter continues to project the expected motion path for up to 45 frames. If a matching bounding box reappears near the predicted area, the trajectory resumes smoothly under the same ID.
* **Camera Overlap Errors (Co-presence Contradictions):** In overlapping fi