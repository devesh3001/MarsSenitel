# Engineering changelog

Corrected after the folder audit. Earlier narratives are preserved in outputs/history_before_repair and the chronological decision log; unsupported claims in them are superseded here. Each entry links an observed symptom, a proposed explanation, a controlled change and its outcome.

### v1: Baseline

I trained a 128-dimensional MSE autoencoder from scratch. Validation MSE was 0.002694; fine texture remained blurred.

### v2 and v3: Loss and capacity

Adding SSIM raised validation SSIM from 0.6611 to 0.6717 with a small MSE cost. Increasing the vector to 256 gave MSE 0.002599 and SSIM 0.6738. More parameters improved reconstruction modestly, not verified anomaly accuracy.

### Threshold and repetition

The initial skew-adjusted fence exceeded every score. I retained it, inspected the multimodal distribution and adopted a fixed mixture envelope. Increasing the forest to 2,000 trees improved repeatability. Independent v3 runs still had flag Jaccards 0.150 and 0.294.

### v4 and v5: Contrast and footprint

Normalization reduced a fixed illumination response, but selected borders became prominent. Excluding connected exact zeros left near-black fragments and weakened forest agreement. I did not promote these variants.

### v6: Near-black border fill

I tested a fixed near-black mask and nearest-valid-pixel fill. Border response decreased, but defect response and forest stability failed the predefined criteria. Filled regions introduced stretched stripes. This was a completed negative experiment, not an abandoned variance mask.

### v7: Gradient loss

I added 0.1 times adjacent-pixel gradient error to 0.9 MSE + 0.1 (1-SSIM). MSE improved 1.68% and gradient error 0.52%. Four of five selected crops were still extremely bright; the darker crop also contains borders. The brightness confound was not solved. The protocol's 0.8 MSE description differed from the actual 0.9 implementation; the saved configuration and source define the executed experiment.

### v8: Geometric augmentation and optional fusion

I added batch flips and quarter-turn rotations to the v7 objective. Validation reconstruction worsened and image-only screening produced zero flags. Lower effective rank does not prove a better representation. A separate post-encoding metadata experiment is reported with its own threshold; zero flags are retained without manufacturing five examples.

### Final decision

I keep v3 as the reference, disclose its brightness and independent-run limitations, and retain all fourteen completed runs as evidence. No hidden-label accuracy is claimed.

## Evidence and limits

Run configurations, histories and diagnostics are included for all fourteen runs. Source splits reduce shared-observation leakage; hidden labels remain unavailable. The canonical notebook and report are the current presentation artifacts.
