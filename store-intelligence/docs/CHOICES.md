# Design Trade-offs & Justifications: Store Intelligence Platform

This document outlines the core technical justifications and architectural trade-offs made during the development of the Store Intelligence Platform.

## 1. Feature Extraction: Color-Spatial Histograms vs. Heavy Deep Networks (CLIP / OSNet)

The system utilizes an optimized, multi-channel color-spatial histogram descriptor within the `FeatureExtractionBackbone` rather than embedding heavy vision networks like CLIP, OSNet, or FastReID directly into the active inference loop.

* **Contextual Adequacy:** The target video clips are short, 2-minute validation sequences for a single retail store. Within this compressed timeline, structural environmental parameters like room lighting and visual background structures are perfectly static. A high-resolution color-spatial descriptor captures uniform apparel configurations reliably.
* **Computational Footprint:** Deep learning models require processing every cropped bounding box through a secondary neural network forward pass sequentially. If 12 individuals inhabit the floor, this adds 12 network inferences per frame step. This approach saturates hardware resources and drops processing speed below the required 15fps threshold. The vectorized histogram engine processes data in sub-millisecond speeds on a basic CPU thread, freeing up GPU resources for the core batched detection model.
* **Dependency Constraints:** Relying on heavy external model backbones balloons the deployment image footprint to over 8GB and requires complex CUDA container configurations. Using a native Python/NumPy feature extractor keeps the Docker container image compact, stable, and highly portable across different evaluation server setups.

---

## 2. Assignment Tracking: Vectorized Global Hungarian Logic vs. Local Greedy Selection

The localized tracking core and cross-camera ReID manager utilize the Hungarian Optimization Algorithm (`scipy.optimize.linear_sum_assignment`) over greedy proximity matchers.

* **Global Optimization:** A local greedy matcher maps tracks independently based on isolated best-match metrics. In crowded spaces, this creates collision risks where adjacent identities attempt to claim the same bounding box, causing frequent identity swaps. The Hungarian algorithm treats the scene as a unified cost matrix, minimizing assignment costs globally to preserve tracking integrity.
* **Multi-Variable Cost Fusing:** Cross-camera matching does not rely purely on visual appearance similarity. The optimization cost function combines appearance scores, temporal time steps, and directional camera transition constraints:
    $$\text{Total Cost} = \text{Appearance Cost} + (0.3 \times \text{Topology Penalty}) + (0.2 \times \text{Temporal Penalty})$$
    This mathematical layout ensures the system safely separates visually similar customers standing in completely different areas of the store.

---

## 3. Analytical Engine: Thread-Safe In-Memory Ledger vs. External Relational Databases

The backend API utilizes an in-memory storage engine protected by explicit re-entrant locks (`threading.Lock`) rather than routing incoming events straight to external disk-backed relational databases like PostgreSQL.

* **Throughput Performance:** The edge pipeline streams structured JSON entries at high frequencies. Forcing a disk-backed database connection to handle write operations on every individual event creates an I/O bottleneck that slows down the network layer. In-memory data structures provide microsecond write access speeds to maintain high data ingestion rates.
* **Concurrency Protections:** Standard dictionary arrays are prone to race conditions and memory corruption when accessed concurrently. Wrapping the write pathways inside a explicit mutual exclusion lock protects internal data states from corruption when web server worker pools handle simultaneous requests.
* **Memory Management:** To ensure long-term stability, the storage module implements an automated data retention check. It sweeps memory logs dynamically based on frame indices, dropping inactive historical records that exceed a 450-frame limit to maintain a bound and predictable memory footprint.

---

## 4. Financial Joins: Symmetrical Sliding Windows vs. Strict Sequential Timelines

To compute the storefront conversion rate, the analytics module runs a symmetrical sliding time window ($\pm 5$ minutes) against the transaction log file (`Brigade_Bangalore_10_April_26 (1)bc6219c.csv`) instead of enforcing a rigid sequential check.

* **Clock Drift Insulation:** Surveillance infrastructure servers and Point of Sale cash registers operate on separate hardware layers. Without a shared master clock, their internal system timestamps can drift apart by minutes. A symmetrical sliding window accommodates this drift gracefully, preventing data dropouts.
* **Process Latency Margins:** Transactions require real-world processing time: scanning products, activating discount codes, processing credit card transactions, and printing invoices. A consumer may exit the checkout queue area several minutes before the payment system logs the final transaction. A sliding window accounts for these manual latency delays, ensuring the conversion metrics remain highly accurate.