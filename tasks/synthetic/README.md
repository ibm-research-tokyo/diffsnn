# Experiments using a synthetic data set

## Set up

```bash
docker build -t diffsnn .
```

## Reproduce Experiment 1
Run
```bash
docker run -v $PWD:/home/docker/tasks -t diffsnn python main.py AnalyzeGradientVariance --working-dir /home/docker/tasks/exp1_diffsnn
docker run -v $PWD:/home/docker/tasks -t diffsnn python main.py AnalyzeGradientVariance --working-dir /home/docker/tasks/exp1_posnn
```
and the variance will be shown in `exp1_[diffsnn/posnn]/ENGLOG/engine.log`.
`diffsnn` for the proposed method, and `posnn` for the existing method.


## Reproduce Experiment 2

Run
```bash
docker run -v $PWD:/home/docker/tasks -t diffsnn python main.py PlotTestLoss --working-dir /home/docker/tasks/exp2 --workers [# of CPU cores]
```
and the experimental result will be created under `exp2/OUTPUT/PlotTestLoss`.


## Reproduce Experiment 3

Run
```bash
docker run -v $PWD:/home/docker/tasks -t diffsnn python main.py PlotTestLoss --working-dir /home/docker/tasks/exp3 --workers [# of CPU cores]
```
and the experimental result will be created under `exp3/OUTPUT/PlotTrainTime`.
