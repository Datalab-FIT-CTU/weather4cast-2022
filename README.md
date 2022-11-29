# WeatherFusionNet: Predicting Precipitation from Satellite Data

![Model diagram](/images/model-diagram.png "Model diagram")

This is our solution to the Weather4cast 2022 competition, we achieved 1st place in the Core Challenge. For more info about the competition, description of the used data and a baseline starter kit, please see [iarai/weather4cast-2022](https://github.com/iarai/weather4cast-2022).

## Usage instructions

- Download the data (see link above) and extract it into the `data` subfolder, or edit `models/configurations/config.yaml` to point to the right folder.
- Install dependencies with
  ```
  conda env create -f environment.yml
  conda activate weather4cast
  ```
- Download trained weights from [Releases](https://github.com/Datalab-FIT-CTU/weather4cast-2022/releases).

### Generating a submission

The `predict-submission.py` script generates a submission zip file for a given challenge and split. For example:
```
python predict-submission.py --challenge core --split test --gpus 0
```
or
```
python predict-submission.py --challenge transfer --split heldout --gpus 0
```
You can find the result in `submission/submission.zip`.

See `python predict-submission.py --help` for more info on the arguments.

