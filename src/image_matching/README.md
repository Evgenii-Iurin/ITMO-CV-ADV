### SIFT + FLANN

- **Description:** After downloading a dataset from Roboflow, it was discovered that some training images had leaked into the test set as augmented versions.
- **Task:** Identify and remove "extra" or duplicate images from the test set.
- **Solution:**
  - Used image hashing to quickly eliminate obviously dissimilar images. This significantly reduced computation before the next steps.
  - Extracted descriptors using **SIFT**
  - Matched descriptors using **FLANN** (Fast Library for Approximate Nearest Neighbors)
  - Analyzed the distribution of match thresholds to determine the optimal cutoff for filtering duplicates
