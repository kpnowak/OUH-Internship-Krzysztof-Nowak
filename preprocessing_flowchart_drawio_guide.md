# Draw.io Flowchart Recreation Guide

## Preprocessing Pipeline Flowchart for draw.io

### Shapes and Colors:
- **Start/End nodes**: Rounded rectangles, light blue (#e1f5fe), dark blue border (#01579b)
- **Clinical Processing**: Rectangles, light purple (#f3e5f5), purple border (#4a148c)
- **Modality Loading**: Rectangles, light green (#e8f5e8), green border (#2e7d32)
- **Processing**: Rectangles, light orange (#fff3e0), orange border (#e65100)
- **Validation**: Rectangles, light red (#ffebee), red border (#c62828)

### Node Layout (Top to Bottom):

#### Row 1: Start & Clinical Processing
```
[📊 Start] → [🏥 Load Clinical] → [📋 Parse Strategies] → [✅ Separators] → [🔧 Repair] → [🎯 Extract Outcome]
```

#### Row 2: ID Standardization
```
[🔗 Standardize IDs] → [📝 Hyphen Format]
```

#### Row 3: Parallel Modality Loading
```
[🧬 Load Modalities] → [📁 Gene Expression] → [🔍 Discovery] → [📖 Reading] → [⚖️ Validation] → [🔗 ID Std] → [🧬 Preprocessing]
                    → [📁 miRNA] → [🔍 Discovery] → [📖 Reading] → [⚖️ Validation] → [🔗 ID Std] → [🧬 Preprocessing]
                    → [📁 Methylation] → [🔍 Discovery] → [📖 Reading] → [⚖️ Validation] → [🔗 ID Std] → [🧬 Preprocessing]
```

#### Row 4: Sample Management
```
[🎯 Sample Intersection] → [🔍 Find Common] → [🔄 Fuzzy Recovery] → [📊 Class Distribution]
```

#### Row 5: Advanced Preprocessing
```
[🔧 Advanced Pipeline] → [⚖️ Modality Scaling]
```

#### Row 6: Scaling Options
```
[❌ Methylation: No Scale] → [✂️ Outlier Clipping]
[📈 Gene: RobustScaler] → [✂️ Outlier Clipping]
[🧬 miRNA: RobustScaler] → [✂️ Outlier Clipping]
```

#### Row 7: Clipping Types
```
[📈 Expression: ±5 SD] → [🎯 Adaptive Selection]
[🧬 miRNA: ±4 SD] → [🎯 Adaptive Selection]
[🔢 Others: ±6 SD] → [🎯 Adaptive Selection]
```

#### Row 8: Feature Selection
```
[📏 Calculate Target] → [🧬 Methylation: f_classif] → [📊 MAD Filtering]
                     → [📈 Gene/miRNA: mutual_info] → [📊 MAD Filtering]
```

#### Row 9: MAD Processing
```
[📊 Calculate MAD] → [⚖️ Scale 1.4826] → [➕ Completeness] → [🎯 Select Features]
```

#### Row 10: Quality Validation
```
[🔍 Quality Validation] → [✅ Orientation] → [✨ Final Data]
                        → [🔢 Stability] → [✨ Final Data]
                        → [🎯 Alignment] → [✨ Final Data]
                        → [📊 CV Integrity] → [✨ Final Data]
```

#### Row 11: Output
```
[✨ Final Data] → [📤 Return Results]
```

### Subgroups (Containers):
1. **Clinical Data Processing**: Contains B, B1, B2, B3, C, D, D1
2. **Modality Loading (Parallel)**: Contains E, E1-E3, F1-F3, G1-G3, H1-H3, I1-I3, J1-J3
3. **Sample Management**: Contains K, K1, K2, K3
4. **Modality-Aware Scaling**: Contains M, M1-M3, N, N1-N3
5. **Feature Selection**: Contains O, O1-O3, P, P1-P4
6. **Quality Assurance**: Contains Q, Q1-Q4

### Connection Types:
- Use straight arrows for sequential flow
- Use curved arrows for parallel branches merging
- Group related nodes in colored containers/subgraphs

### Text Formatting:
- Use emojis at the start of each node label
- Break long text with line breaks (Shift+Enter in draw.io)
- Keep text concise but descriptive 