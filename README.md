diabetic  retinopathy detection

use a dataset  that was found in my report

kinda copy the processs used  in my report

train it on 3 datasets (80%/20% split so train is 80% of all and evaluate is 20% of all)
- EyePACS (lowkey skip bc 88gb)
- APTOS 2019
- MESSIDOR

slice images into bags

use optuna on a small sample of the dataset (proxy tuning )to find parameters for full training (color spaces, blurs, etc)

maybe circular cropping (trial it I guess)

fixed parameter pre-processing (using optuna found settings)

run  this baby through a CNN (try denseNet51 and  resNet112 or whatever they used in the report.  I forgot  the numbers tbh)

slice the images into bags and run them thorugh cnn + attention mechanism

