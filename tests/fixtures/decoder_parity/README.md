# Decoder-parity fixture

`Te-no_120.jpg` is from the test split of the *Brain Tumor MRI Dataset* by
Masoud Nickparvar (Kaggle), licensed CC BY 4.0. It is one of the JPEGs that
Pillow 13 decodes differently from Pillow 10.3/12.3. The upgrade to 13.0.0 was
made only after measuring that it changes no prediction on 2,441 test and
validation images (results/metrics/decoder_parity.json); the golden-hash test
fails if a future dependency upgrade silently changes model input again.
