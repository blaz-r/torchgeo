# TorchGeo Workshop — Attendee Guide

Two 75-minute sessions, one coffee break. By the end you will have trained a change-detection model on real Sentinel-2 imagery.

---

All notebooks live in `docs/tutorials/`. Open them in Jupyter or VS Code.

---

## Session 1 — Intro to TorchGeo (75 min)

**Theme**: *What makes geospatial data hard, and how TorchGeo solves it.*

### Why geospatial ML is different → `geospatial.ipynb`

- Data comes in many modalities: multispectral, SAR, LiDAR
- Each file may have its own CRS, resolution, and extent
- You can't just hand GeoTIFFs to a standard DataLoader

### PyTorch refresher → `pytorch.ipynb`

- Dataset → DataLoader → CNN → train loop → eval
- Uses `EuroSAT100`: 100 Sentinel-2 RGB patches, 10 land-cover classes

### GeoDataset (the core concept) → `torchgeo.ipynb` ⭐

This is the most important part of session 1. TorchGeo's `GeoDataset`:

- Indexes every file by its spatiotemporal bounding box
- **Union** (`|`): merge overlapping scenes into a virtual mosaic
- **Intersection** (`&`): sample only where inputs and labels overlap
- Reprojects and resamples on the fly — no preprocessing step
- Works with `RandomGeoSampler` (training) and `GridGeoSampler` (evaluation)

```python
landsat = landsat7 | landsat8          # mosaic
dataset = landsat & cdl                # paired inputs + labels
sample  = dataset[xmin:xmax, ymin:ymax, tmin:tmax]  # windowed read
```

### Transforms and spectral indices → `transforms.ipynb`, `indices.ipynb`

- `MinMaxNormalize`, chained augmentation pipelines
- NDVI, NDWI, NDBI computed directly from band tensors
- Kornia for GPU-accelerated augmentation in production

### Pretrained weights + trainers (preview) → `pretrained_weights.ipynb`, `trainers.ipynb`

- Multi-weight API: inspect `weights.meta` to see what data a model was pretrained on
- `ClassificationTask` + Lightning `Trainer`: the pattern you will use in session 2

---

## ☕ Coffee break

---

## Session 2 — Hands-On Use Cases (75 min)

**Theme**: *Four real workloads, building up to a trained change-detection model.*

### Pretrained inference, no training → `naip_road_segmentation.ipynb`

- Load a 4-band NAIP tile directly from a remote URL (no download)
- Load `Unet_Weights.NAIP_RGBN_RESNET18_CHESAPEAKERSC` and its preprocessing transforms
- Run inference and compare predicted road mask against ground truth
- Takeaway: *this is the minimum-effort TorchGeo workflow*

### Two approaches to the same problem → `earthquake_detection.ipynb`

Uses `QuakeSet`: Sentinel-1 SAR image pairs (pre/post earthquake) with binary labels.

**Approach 1 — Classical ML on frozen embeddings**
1. Load `ResNet50_Weights.SENTINEL1_GRD_MOCO` (pretrained on Sentinel-1)
2. Extract embeddings for every sample
3. Fit a `RandomForestClassifier` on the embeddings

**Approach 2 — Fine-tune a deep model end-to-end**
1. `ClassificationTask(model='resnet18', in_channels=4, task='binary', loss='bce')`
2. Lightning `Trainer` with `limit_*_batches` for a quick run

When to use which: classical ML is faster and needs fewer labels; DL wins when you have enough data and need custom features.

### Bring-your-own GeoTIFFs → `custom_raster_dataset.ipynb`

TorchGeo has ~70 built-in datasets. For your own files, subclass `RasterDataset` with a few class attributes:

```python
class Sentinel2(RasterDataset):
    filename_glob = 'T*_B02_10m.tif'
    filename_regex = r'^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])'
    date_format    = '%Y%m%dT%H%M%S'
    is_image       = True
    separate_files = True
    all_bands      = ('B02', 'B03', 'B04', 'B08')
    rgb_bands      = ('B04', 'B03', 'B02')
```

That's it — `GeoDataset` handles indexing, CRS alignment, and windowed reads automatically.

> **Note**: the download step in this notebook uses the Planetary Computer API (requires a Microsoft account). The rest of the notebook runs fine on pre-downloaded files.

### Change detection end-to-end → `change_detection.ipynb` 🎯

Uses `OSCD100`: 100 Sentinel-2 image pairs from 14 cities, binary change mask.

**Key configuration choices:**
| Parameter | Value | Why |
|---|---|---|
| `model` | `'btc'` | BTC uses separate encoders per timestep + temporal fusion |
| `backbone` | `'swin_tiny'` | Swin Transformer pretrained on Cityscapes |
| `freeze_backbone` | `True` | Only fine-tune the decoder — faster convergence on small data |
| `pos_weight` | `10.0` | Most pixels are no-change; upweight the rare change class |
| `loss` | `'bce'` | Binary cross-entropy for binary segmentation |

**What to expect after training (~50 s on GPU):**
- Accuracy: ~93–95%
- F1 Score: ~0.55–0.65
- IoU: ~0.40–0.50

### Wrap-up

The pattern from session 1 (`DataModule` + `Task` + `Trainer` + callbacks) is identical across every use case. Once you know it, any TorchGeo workflow looks familiar.

---

## Going Further

| Notebook | What it shows |
|---|---|
| `custom_segmentation_trainer.ipynb` | Override `configure_optimizers` / `configure_metrics` for custom training loops |
| `cli.ipynb` | YAML-driven training — same workloads, no Python code required |
| `contribute_*.ipynb` | How to contribute a dataset or DataModule back to TorchGeo |

Full model and weights catalog: [torchgeo.readthedocs.io](https://torchgeo.readthedocs.io)
