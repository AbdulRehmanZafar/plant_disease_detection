# samples/ — your demo images

Put 8–10 leaf photos here **before the exhibition**, named by their true class,
so you can demo instantly without browsing the dataset:

```
tomato_early_blight_01.jpg
tomato_healthy_01.jpg
potato_late_blight_01.jpg
potato_early_blight_01.jpg
corn_common_rust_01.jpg
grape_black_rot_01.jpg
apple_scab_01.jpg
pepper_bacterial_spot_01.jpg
strawberry_healthy_01.jpg
```

## How to fill it in 2 minutes

After running `prepare_dataset.py`, copy one image from each of the classes you
want to demo:

```powershell
# one image per class, copied out of the TEST split (never trained on)
$split = "data\plantvillage_split\test"
Get-ChildItem $split -Directory | ForEach-Object {
    $first = Get-ChildItem $_.FullName -File | Select-Object -First 1
    Copy-Item $first.FullName "samples\$($_.Name)_$($first.Name)"
}
```

## Why test-split images matter

Images from the **test** split were never used for training. If they classify
correctly, that is genuine generalisation and you can tell the examiner:
"this image was held out — the model has never seen it."

Images from the **train** split prove nothing, and if the examiner spots it, it
looks like you are hiding something.

## Also good to have

- 2–3 photos you took yourself with a phone (even of a houseplant) — real-world
  demonstration beats any curated sample.
- one image that is **not** a leaf (a pen, a shoe) for the "closed-set
  limitation" answer. Script: see `DEMO_SCRIPT.md` section D.
