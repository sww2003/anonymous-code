# Dataset layout

Dataset contents are intentionally excluded from version control. The default
configuration expects:

```text
datasets/
├── train/HR/
└── RealSR/
    ├── LR/
    └── HR/
```

Use relative paths in experiment configurations. Do not commit dataset files,
generated LMDB databases, or machine-specific mount paths.
