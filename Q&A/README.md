# General Q&A
#### Question. Test, Train Set의 할당에 황금(?) 비율이 있는지요? 그리고 예를 들어 data가 충분하지 않을때 Train에 많이 할당하면 좋을텐데, 이런 경우를 위해서 할당량을 조절할 방법이 있나요?

    train / test split의 목적은 generalization error 를 최소화 하기 위함인데, 둘 사이의 샘플 개수 결정은 경쟁 관계에 있습니다. 
    만약 train 사이즈가 작으면 paramter estimate의 variance 가 높고, 반대로 test 사이즈가 작으면 performace statistic의 variance가 높아집니다. 
    따라서, 양쪽의 variance가 너무 높지 않도록 설정해야하는데, 이는 sample size에 따라서 그 크기를 결정하는 방식에 차이가 날 것 같습니다.

    예를들어 100개의 sample에 대해서 train / test 를 쪼갤때는 어떤 방식을 취해도 parameter estimate variance를 줄이기가 쉽지가 않을 것이고, 100000개의 sample을 train / test 를 쪼갤때는 0.8:0.2 를 취하거나, 
    0.9:0.1 을 취하든 generalization에 문제가 없을 것으로 보입니다. 
    어떤 황금률을 제시하기는 어렵지만, variance를 줄이기 위한 방법은 아래와 같은 방법이 있습니다.

        1. training 데이터를 또 다시 train / validation으로 쪼개는 방법
        2. train을 비율 (90, 80, 70) 등을 바꿔가면서 각 비율당 여러차래 (ex. 20회 이상) random sampling 하여 mean performance를 비교하여 lower variance를 선택하는 방법

    저는 약식으로 할때는 보통 70/30 혹은 80/20을 선호합니다. 

    이론적인 부분을 조금 더 살펴보고 싶으시면 아래 논문을 추천드립니다.
    https://pdfs.semanticscholar.org/452e/6c05d46e061290fefff8b46d0ff161998677.pdf


#### Question. data을 scaling하고 train 하는 이유가 있을것 같아서 문의드립니다.  어떤 모델에서든 항상 0~1값으로 변경해야하는것인지?

    이론적인 면에서는 Normalize는 Neural Network의 Weight, Bias가 조정되는 것으로 모두 잡힐 수 있어야 하지만, Practical 하게는 
        1. training more stable and faster 
        2. reduce the chances of getting stuck in local optima 
        3. 학습과 예측에 사용되는 Domain 데이터 사이의 샘플링 Population 일치 문제 (out of scale from one another)
    를 해결 하기 위한 것으로 알려져 있습니다.

    1번, 2번의 경우를 보다 자세히 얘기드리자면, gradient descent를 할때는 gradient error vectors와 learning rate를 곱해서 weight를 조정해 주는데, 
    input feature들 간의 distribution 차이가 나면, 한쪽 feature에 대해서는 잘 작동하는 learning rate가 다른 feature에 대해서는 under compensating하는 문제가 있을 수 있습니다. 
    (since the learning rate in the update equation of Stochastic Gradient Descent is the same for every parameter) (Elkan, Charles. (2008) Log-linear models and conditional random fields)

    이런 이유로 error surface에서 Global minimum을 찾기 위해서는 구모양에 가까운 input feature를 가지는 것이 타원형 구조 (한쪽 feature의 range가 훨씬 큰 경우)보다 빠르게 converge 하는 것으로 알려져 있습니다.

    3번 같은 경우에 조금 더 쉬운 예를 들자면, 주식/코인 같은 경우에는 Price를 그대로 입력하는 것이 아니라, 
    Return, Log Return을 Scaling의 목적으로 사용할 때가 있는데, 학습할때의 Price 값 (ex: KOSPI: 2000, BTC: $3000)의 기준과 예측할 때(ex: KOSPI: 2300, BTC: $7000)의 Price 값의 기준이 크게 차이가 나는 문제를 해결하기 위해서 사용합니다. 
    (같은 input feature이지만, 시기가 달라서 학습에 관찰되지 않은 데이터가 입력 될 수 있음, out of scale from one another) 

    또한, 가격에 대한 입력과, Bid/Ask Volume에 대한 입력은 Scale 차이가 상당한데 이러할 경우에, 모든 feature에 기반하여 모델이 작동하기 보다는,
    Range가 보다 큰 particular한 feature에 모델이 종속되어 버리는 것 같습니다.  
    (ex. 모델 내에서 distance metric이 계산될 경우 값의 범위가 큰 feature에 종속)

    그리고, 마지막으로 0~1 값으로 변경해야 할지, 다른 옵션 (Mean Normalization, Standardization, Scaling to unit length)를 할지, 
    혹은 Scailing이 하지 않을지 등에 대한 Feature Scaling 선택 문제가 있을 수 있는데, 
    풀고자 하는 문제의 특성에 따라서 다른 Scaling 방법을 취해주는 것이 좋을 것 같습니다. 
    가량 Exact Boundary가 있는 문제의 경우에는 Standardization이 적절하지 않을 수도 있고, 
    min-max scaling은 long tail 분포가 있는 문제에 대해서는 적절하지 않을 수 있습니다. 

#### Question. 텐서플로우에서 모델 저장하면 checkpoint 파일등 여러개가 만들어지잖아요. 케라스에서는 .h 파일 하나만 생기는겁니까? 혹시 텐서플로우의 .pb 파일에 대응하는 건가요?

    Checkpointing 기능은 긴 시간 (days, weeks) 동안의 학습시, train과정에 문제가 있더라도 모든 학습 정보를 잃지 않도록하는데 필수적인 기능이여서, 장기적으로는 꼭 배워둘 필요가 있는 것 같습니다. 

    질문 주신 Keras의 경우에는 아래의 callback API를 통해 checkpointing 기능을 제공합니다. 
    해당 기능을 통해서, 저장 위치, checkpoint 이름 (epoch number 혹은 metric으로도 naming 가능), 
    어떠한 조건 (save_best_only, save_weights_only, period)을 만족하는 checkpoint를 생성 할지를 설정할 수 있습니다.
    https://keras.io/callbacks/#modelcheckpoint

    Google Colab에서는 얘기주신 것처럼 training 이후 HDF5 format으로 전체 모델을 직접 저정할 수도 있으며, 
    callback API를 통해 training 과정 중에 모델을 저장하는 것도 가능합니다. 
    자세한 checkpoint 저장 방법은 아래의 튜토리얼을 참조해 주세요. 
    https://colab.research.google.com/github/tensorflow/models/blob/master/samples/core/tutorials/keras/save_and_restore_models.ipynb#scrollTo=2S4xrNJRilwi

#### Question. keras에서 Build Model부분이 마치 black box 같아 보입니다. 모델 학습에도 변수들의 변화되는 과정을 print 해볼만한 방법이 있을까요? 

    neural net의 디버깅은 원인이 너무나 다양할 수 있어서 (exploding gradients, dead relu, vanishing gradient etc) 항상 쉬운 일이 아닌것 같습니다. 
    저는 주로 model compile 이후에 "model.summary()" 를 통해 전체 shape 구조를 보거나, keras model의 특정 layer를 취득하여 해당 layer의 결과를 확인할 수 있는 function을 만들어 사용합니다. 
    해당 방법에 대해서는 아래 링크를 참조해 주시면 됩니다.

    http://laid.delanover.com/debugging-a-keras-neural-network/

    추가적으로 neural network 모델의 일반적인 디버깅 방식은 아래의 링크를 참조해 주시면 좋을 것 같습니다.
    https://medium.com/machine-learning-world/how-to-debug-neural-networks-manual-dc2a200f10f2

#### Question. Colab에서 with tf.device('/gpu:0') 아래에 인덴트해서 넣으면 CPU가 아닌 GPU로 돌아가는 게 맞는지요? 

    안녕하세요 :slightly_smiling_face: 좋은 질문 주셔서 감사합니다. 상단 메뉴의 런타임에서 런타임 유형을 꼭 GPU로 바꿔주셔야 GPU가 작동됩니다. 
    아래 링크에서 참조해 보시면, GPU 설정 이전에는 with tf.device('/gpu:0)을 설정하더라도, 
    GPU 장치가 관측되지 않습니다. (https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=t9ALbbpmY9rm)

#### Question. 딥러닝에서 레이어를 몇개로 하며 또 각 레이어마다 차원을 몇개로 해야하는지는 어떻게 정하는 겁니까?

    안녕하세요. 좋은 질문 주셔서 감사합니다. 딥러닝에서 레이어의 개수 혹은 차원의 개수에 대한 결정은 Overfitting Issues와 큰 관련이 있습니다. 
    Overfitting은 1. 모델이 너무 복잡하거나 2. 데이터가 너무 적거나 하는 문제로 인해서 발생할 수 있습니다.

    1번의 모델이 너무 복잡해서 발생하는 문제는 주로 Cross Validation을 통해 Overfitting이 발생하지 않았는지를 테스트하며, 같은 결과면 간단한 모델이 더 좋은 모델 (오캄의 면도날)의 원칙을 적용하는 것을 권장합니다. 
    관련한 보다 자세한 이슈는 아래 링크를 통해 공유드립니다. (https://drive.google.com/open?id=1HqCyp94ZC9f7jwhStFnSbuF9TDyrf5iy) 
    Overfitting을 해결하기 위한 방법은 Module3에서 관련하여 보다 자세히 다룰 예정입니다. 
    (Learning Curves, Batch Normalization, Dropout and Regularization, Continuous Learning, Hyperparameter Search)

    2번의 데이터 샘플에 대한 이슈는, 주어진 데이터에 대해 Train 데이터, Test 데이터의 크기를 결정해야 할 때가 있는데, 둘 사이의 샘플 개수 결정은 경쟁 관계에 있습니다. 
    최종적인 목적은, generalization error 를 최소화 하기 위함인데, 만약 train 사이즈가 작으면 paramter estimate의 variance 가 높고, 
    반대로 test 사이즈가 작으면 performace statistic의 variance가 높아집니다. 
    따라서, 양쪽의 variance가 너무 높지 않도록 설정해야하는데, 이는 sample size에 따라서 그 크기를 결정하는 방식에 차이가 날 것 같습니다.

    예를들어 100개의 sample에 대해서 train / test 를 쪼갤때는 어떤 방식을 취해도 parameter estimate variance를 줄이기가 쉽지가 않을 것이고, 
    100000개의 sample을 train / test 를 쪼갤때는 0.8:0.2 를 취하거나, 0.9:0.1 을 취하든 generalization에 문제가 없을 것으로 보입니다. 
    variance를 줄이기 위한 방법으로는 train을 비율 (90, 80, 70) 등을 바꿔가면서 각 비율당 여러차래 (ex. 20회 이상) 
    random sampling 하여 mean performance를 비교하여 lower variance를 선택하는 방법을 쓰기도 합니다.

    Cross Validation에 대한 설명은 아래 링크를 참조해 보시면 좋을것 같습니다. (https://www.youtube.com/watch?v=TIgfjmp-4BA)
    모델이 복잡할때 생길 수 있는 이슈에 대한 예제는 아래의 링크를 참조해 보시면 됩니다. (https://towardsdatascience.com/beginners-ask-how-many-hidden-layers-neurons-to-use-in-artificial-neural-networks-51466afa0d3e)

#### !pip install -q matplotlib-venn 이 뭐하는건지는 알겠는데, -q는 뭘까요?

    아래 링크에 따르면, -q는 설치 과정에 나타나는 메세지를 줄여주는 역할을 합니다.

    -q, --quiet

    Give less output. Option is additive, and can be used up to 3 times (corresponding to WARNING, ERROR, and
    CRITICAL logging levels).

    https://media.readthedocs.org/pdf/pip/latest/pip.pdf

#### hyper parameter 최적화를 위한 설정 (Work-in-Progress)

#### Keras와 Google Coolab을 사용하는데 가장 적절한 파이썬 버전은 무엇인가요? (Work-in-Progress)

#### AI로 뭔가를 학습 시킬 때 전처리를 할텐데 이런 타임시리즈의 데이타는 어떤식으로 전처리를 하나요? 이동평균선?, 볼린져밴드 등 사용하나요? (Work-in-Progress)

