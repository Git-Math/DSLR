# DSLR
Project to learn data science and logistic regression:
- Read a data set, visualize it in different ways, select and
clean unnecessary information from data.
- Train a logistic regression from scratch that will solve classification problem.

## Data analysis
Display informations about a given dataset.

### Example
```
$ python3 describe.py resources/dataset_train.csv
                    Arithmancy           Astronomy           Herbology Defense Against the          Divination      Muggle Studies       Ancient Runes    History of Magic     Transfiguration             Potions Care of Magical Cre              Charms              Flying
     Count       1566.00000000       1568.00000000       1567.00000000       1569.00000000       1561.00000000       1565.00000000       1565.00000000       1557.00000000       1566.00000000       1570.00000000       1560.00000000       1600.00000000       1600.00000000
      Mean      49634.57024266         39.79713089          1.14101953         -0.38786350          3.15390967       -224.58991486        495.74797006          2.96309462       1030.09694639          5.95037299         -0.05342714       -243.37440901         21.95801250
       Std      16679.80603556        520.29826761          5.21968199          5.21279371          4.15530090        486.34483965        106.28516458          4.42577466         44.12511587          3.14785425          0.97145697          8.78363988         97.63160207
       min     -24370.00000000       -966.74054564        -10.29566275        -10.16211940         -8.72700000      -1086.49683489        283.86960873         -8.85899299        906.62731969         -4.69748377         -3.31367576       -261.04892000       -181.47000000
       25%      38511.50000000       -489.55138715         -4.30818185         -5.25909540          3.09900000       -577.58009634        397.51104693          2.21865332       1026.20999269          3.64678504         -0.67160647       -250.65260000        -41.87000000
       50%      49013.50000000        260.28944642          3.46901210         -2.58934161          4.62400000       -419.16429373        463.91830515          4.37817554       1045.50699629          5.87483735         -0.04481105       -244.86776500         -2.51500000
       75%      60811.25000000        524.77194897          5.41918348          4.90468008          5.66700000        254.99485732        597.49222952          5.82524166       1058.43641045          8.24817254          0.58991935       -232.55230500         50.56000000
       max     104956.00000000       1016.21194039         11.61289508          9.66740546         10.03200000       1092.38861057        745.39621988         11.88971275       1098.95820054         13.53676212          3.05654577       -225.42814000        279.07000000
       Var  278215929.38388073     270710.28727294         27.24508011         27.17321824         17.26652555     236531.30305619      11296.53620947         19.58748131       1947.02585026          9.90898638          0.94372864         77.15232947       9531.92972238
 Ravenclaw      49446.34570766       -480.04308342          5.02185702          4.81808831          4.98789376        489.11196678        597.74996800          4.91596015       1050.33428225          6.97241318          0.00389914       -231.08057630         -3.97979684
 Slytherin      49374.80677966       -496.33392706         -4.77562437          4.96401644         -4.81449141       -478.73188331        401.81387404          4.96768098       1051.89869816          9.47074327         -0.08734849       -249.59319402        -70.32166113
Gryffindor      49121.99685535        493.33664640         -4.78390586         -4.94800034          4.90134891       -501.48314944        596.93448719         -4.81494744        951.13469615          2.93813244         -0.13918582       -252.73104557        189.02452599
Hufflepuff      50249.04022989        497.70800234          4.91940766         -4.98166599          5.02166473       -498.35077377        400.14745367          5.01029481       1049.56216455          4.96068607         -0.02844939       -244.34736450         -7.08568998
```

## Data visualization
Scripts to visualize data.

### Histogram
`$ python3 histogram.py resources/dataset_train.csv`

![dslr_histogram](https://github.com/Git-Math/DSLR/assets/11985913/93c9603f-4bff-4c08-aecb-255974feb9f4)


### Scatter plot
`$ python3 scatter_plot.py resources/dataset_train.csv`

![dslr_scatter_splot](https://github.com/Git-Math/DSLR/assets/11985913/82c13ad0-407e-4e95-a41d-2f8bd0aa1413)


### Pair plot
`$ python3 pair_plot.py resources/dataset_train.csv`

![dslr_pair_plot](https://github.com/Git-Math/DSLR/assets/11985913/22361eb5-6967-40b8-9e9e-5d5e550c5b17)

## Logistic regression

### Training
Train the model using logistic regression with gradient descent.

```
$ python3 logreg_train.py resources/dataset_train.csv

$ cat theta.csv
house,theta_matrix
Ravenclaw,-11.142174849740677,2.838416021017299,-3.1953740883363047,4.088090164943005,2.5480855167186354,0.10872116184932121,4.008428145561912,3.31877831728751,-1.9182008620001283,-3.0554817270937704,2.07463526509445,0.49827498982330276,7.387037993473109,-0.3179866532419291
Slytherin,0.4586969473846786,-0.8369206785397498,-3.4227494484433443,0.23062810240481574,5.604524269837075,-6.512062865339304,-0.43204059214162355,-0.9760585507377478,-0.29180478803273113,6.61532728799827,3.3477719242942148,-0.992827602765324,-9.656664824489583,-2.3965477502355235
Gryffindor,-4.700655015657594,-9.432113222800774,-0.44649070862996465,-9.868137674190558,1.363185552955173,9.606372459836795,-2.4775151779118776,9.702668881824394,7.73405363455109,0.6836344164666845,-9.172664946589164,0.5224126083316912,-5.6030025735759255,9.714129838486514
Hufflepuff,-0.2025798648552964,3.433627806196958,5.011324533946532,8.016178620787322,-7.284632768918365,1.2773645010045698,-6.626907067592443,-9.010396914197921,-0.8655382444267954,-2.8098340112589026,1.9925851725382202,0.030493199327144914,5.064370939224866,-6.345392401041838
```

### Prediction

#### Logreg predict
Makes prediction on a given dataset with the trained model.

```
$ python3 logreg_predict.py resources/dataset_test.csv theta.csv

$ cat houses.csv
Index,Hogwarts House
0,Hufflepuff
1,Ravenclaw
2,Gryffindor
3,Hufflepuff
4,Hufflepuff
5,Slytherin
6,Ravenclaw
7,Hufflepuff
8,Ravenclaw
9,Hufflepuff
10,Hufflepuff
...
389,Ravenclaw
390,Slytherin
391,Hufflepuff
392,Hufflepuff
393,Ravenclaw
394,Ravenclaw
395,Slytherin
396,Hufflepuff
397,Hufflepuff
398,Ravenclaw
399,Ravenclaw
```

#### Predict house
Prompts grade values and predict the house.

```
$ python3 predict_house.py theta.csv
Fill your marks
Arithmancy: 18
Astronomy: 11
Herbology: 7
Defense Against the Dark Arts: 16
Divination: 13
Muggle Studies: 15
Ancient Runes: 9
History of Magic: 6
Transfiguration: 13
Potions: 7
Care of Magical Creatures: 8
Charms: 10
Flying: 17
Sorting Hat: Hum, you will go to... Ravenclaw!
```
