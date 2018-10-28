# Module1
## Module1.1 Introduction
#### [주간 질문] AI를 통해 Finance의 어떤 문제들을 해결할 수 있을까요? 

    1. Increasing security: security와 관련된 빅데이터 분석을 통한 보안 향상. (Anti-Money Laundering, fraud detection 등등)
    2. Reducing processing times: financial documentation 처리에 걸리는 시간을 줄여주는 효과.
    3. Algorithmic trading:trading pattern을 찾고, trends를 예측하기 위하여 AI를 적용시켜서, 더 성공적인 stock trading decisions을 내리도록 함.
    4. Customer recommendation / Credit lending: consumer data, financial lending 혹은 insurance results등의 자료를 기반으로, 고객들의 관심분야를 추천에서부터, 대출 여부를 평가하는 등의 분야에도 적용 가능함.
    5. Porfolio Management

#### Question1. 머신러닝이 할수 있는 예측에 대하여 궁금합니다. 일부에서는 스포츠 승률이라던지 심지어는 복권당첨도 예측하던데요.

    머신 러닝은 짧은 시간 동안에 여러 분야에 극적인 영향을 끼치고 있습니다. 예측 할 수 있는 문제의 유형도 시간이 지나면서 Combinatorial하게 넓어지고 있는 것 같습니다. 
    승포츠 승률, 복권 당첨 뿐만아니라, 기존에 연구되어 왔던 거의 모든 주제에 있어서 새로운 방법론의 등장으로 인해 새롭게 다루어 지고 있는 것 같습니다.

    개인적으로 제 주변에서 재밌게 하고 있는 연구들은, 
        1. 의료 분야 (쥐의 똥에서 추출한 데이터에 기반해서 쥐의 장 내부에 있는 모든 박테리아의 종류와 양을 예측함), 
        2. 암 환자들의 모바일 사용 정도로 우울증 상태를 예측, 
        3. 어떤 식으로 Speech를 한 벤처 기업가가 성공적으로 투자를받을 수 있을지 예측 
    등등이 있는 것 같습니다. 이외에도 나열하자면 끝이 없는 예측 문제들이 있습니다.

    현실에서의 Application이 궁금하시면, Top 10 Applications of Machine Learning (https://www.youtube.com/watch?v=ahRcGObyEZo)을 참고해 보시면 좋을 것 같습니다.

    혹시 경제학 분야에서의 적용이 궁금하시면, Stanford의 Susan Athey 교수의 "The Impact of Machine Learning on Economics" 논문을 읽어보시길 권유드립니다. (https://www.nber.org/chapters/c14009.pdf)
    
#### Question2. 과연 딥러닝으로 금융시장을 이길수있는 전략을 도출할수있을까요?

    효율적 시장가설에 따르면 알려진 정보로는 수익(alpha)을 낼 수 없습니다. 따라서, 다른 사람에 비해서 Superior한 정보, 방법론 등을 가지고 있는 것이 중요한 것 같습니다. 

    주식에서 주가 예측 모형으로 가장 유명한 Fama French 3 Factor 모델을 예로 들자면,
    해당 모델을 통해서 과거 몇년 동안은 예측률이 있고 수익이 났지만, 해당 모델이 널리 알려진 뒤에는 
    더 이상 다른 사람에 비해 Superior한 모델이 아니기 때문에 수익이 관측되지 않는다 라는 논문이 있습니다.
    
    또한 주식에서는 큰 주식은 이미 너무나 효율적이여서 예측 모델로 수익을 내기가 쉽지 않고, 
    작은 주식은 거래량이 적어서 높은 슬리피지로 인해 수익을 만들어 내기 어려운 점이 있습니다. 
    그래서, 중간 사이즈 크기의 주식에서 오히려 Back test시 수익이 나는 경우도 있는 것 같습니다. 
    코인의 경우도 마찬가지로 BTC에서는 이미 여러 시장 참가자들에 의해 Alpha가 사라지고,
    중간 사이즈의 코인에서의 수익률이 더 나아보기도 합니다. 

    그런데, 예측 모델의 Accuracy Level을 다른 사람의 모델에 비해서 훨씬 더 끌어올릴 수가 있다면, 
    (새로운 Factor 사용, 다른 모델 사용)
    우월한 방법론으로 Alpha를 찾을 수 있지 않을까 기대하고 있습니다.

    하지만, 주의하셔야 할점은 M1.2에서 다루었듯이, Theory 없이 (Causual Relationship) 잘 못된 Factor에 기반한 예측은 심각한 Bias를 만들어 낼 수도 있기 때문에 주의가 필요한 것 같습니다.
    
## Module1.2 Machine Learning for Financial Time Series Prediction
#### [주간 질문] Deep Learning Predictor를 이용한 방법은 Traditional Predictors에 비해 어떤 이점이 있을까요?

    1. casual model보다 예측에 더 나은 성능을 보인다. 
    2. 관련이 있는 모든 데이터들을 포함하도록 input data를 확장시킬 수 있다. 
    3. 선형모델만으로는 표현할 수 없는 것들을 머신러닝은 비선형으로 표현할 수 있음 
    4. over-fitting 문제를 쉽게 피할 수 있음. (Cross validation, Dropout for model selection, Regularization techniques등을 통해)
    5. 가장 좋은 모델을 만들기 위해 머신러닝은 자동적으로 feature를 선택할 수 있다.

#### Question1. 인과추론과 예측의 차이를 예를 들어 설명해주세요. 편향없는 추정과 가장 유사한 상관관계는 어떻게 다른가요

    해당 자료를 통해서 말하고 싶은 바는, Machine Learning이 어떤 문제를 효과적으로 풀 수 있고, 어떤 문제에는 효과적이지 않은지에 대해 다루고 싶었습니다. 
    이는 곧 Financial Time Series Prediction에 있어서 주의해야 할 점이여서 강의 초반부에 집고 넘어가고 싶었고요.

    핵심만을 요약드리지면, Machine Learning을 통해서는 Prediction 문제를 효과적으로 풀 수 있는 반면에 Causation에 대한 주장은 할 수 없습니다. 
    이론적인 배경 없이 어떠한 Factor가 어떠한 결과를 야기했다고 볼 수 없기 떄문에, 이러한 예측은 잘 못된 결과를 만들어 낼 수 있습니다.

    조금 더 구체적으로, Prediction과 Causation의 차이를 단적으로 예를 들면, 만약 전자레인지 폭발에 의해 불이나 난 상황에서 
        1. Causation: 불이 일어난 원인은 전자레인지 폭발에 의한 것입니다. 
        2. Correlation: 소방관이 몰려 있는 곳에는 불이 났을 수 있습니다. 하지만, 소방관이 불을 일으키진 않았고, 
                        둘 사이에는 상관관계가 높을 뿐 입니다. 
        3. Prediction: 하지만 소방관이 몰려있다는 것으로 부터, 불이 났을 거라는 예측을 할 수는 있습니다.

    따라서, Prediction에 있어서는 Causation과 관련 없이 Correlation이 중요 합니다. 
    하지만, 만약 소방관이 몰려 있는 것이 불을 일으킨다고 잘 못 해석을 하면, 치명적인 예측 오류를 범하게 됩니다. 

    Prediction과 Causation은 목표로 하는 역할이 다르며, 저희가 Machine Learning을 통해 풀고자 하는 문제의 목표는 Prediction Level을 높히기 위함입니다. 
    따라서, 이에 따른 단점 (인과성 부족, 이론 부족)을 인지하신 뒤에, 일반적인 인과 관계 추론 모형 (Regression)보다 어떤 점이 더 나을 수 있는지 (비선형성, Over fitting 문제를 더 쉽게 다를 수 있음 등)를 숙지하셨으면 합니다.

#### Question2. 6-10 Page 및 7,8 Page 수식에 대한 추가 설명

    1. 6 page
        - Explanation 모형과 Prediction 모형에 있어서 4가지 큰 차이점을 볼 수 있습니다.
        - Theory – Data
            - Explanation 모형은 Theory에 기반하여 예측 모형의 형태를 신중하게 설정합니다. 예를 들어 선형 관계이면 왜 선형관계인지, Curve Linear 관계를 지니면 왜 그런지에 대해서 다루며, 입력 데이터와 출력 데이터 사이의 관계가 중요합니다.
            - Prediction 모델에서는 예측 함수의 형태는 데이터로 부터 나오며, 입력 데이터와 출력 데이터 사이의 관계는 중요하지 않습니다.
        - Causation – Association
            - Explanation Model에서는 X가 Y를 Cause합니다
            - Prediction Model에서는 X가 Y와 Correlation이 있습니다.
        - Bias – Variance
            - Explanation 모형에서는 Bias를 최소화한 Factor (Causual Effect)를 찾는 문제를 풀지만
            - Prediction에서는 Bias와 Variance의 합을 최소화 하는 모형을 찾습니다.
        - Retrospective (in-sample) – Prospective (out-of-sample)
            - Predictive modeling은 forward-looking을 목적으로 하는 반면에
            - Explanation 모형은 이미 세워진 가설 관계를 검증하는 용도로 사용합니다.

    2. 7 page
        - Bias: Bias is the result of misspecifying the statistical model f.
        - Variance: Estimation variance (the third term) is the result of using a sample to estimate f.

        - The first term is the error that results even if the model is correctly specified and accurately estimated
        - The above decomposition reveals a source of the difference between explanatory and predictive modeling:
            - In explanatory modeling the focus is on minimizing bias to obtain the most accurate representation of the underlying theory. 
            - In contrast, predictive modeling seeks to minimize the combination of bias and estimation variance, occasionally sacrificing theoretical accuracy for improved empirical precision

    3. 8 page
        - Causual Inference는 Unbiased estimates를 목표로 합니다.
        - 이를 검증하기 위해서, 연구 대상 A가 factor X에 영향을 받았을 때와 받지 않았을 때의 차이를 보고 싶은데,
        - Observational Data (실험이 아닌)에 기반하면 사실은 연구 대상 A가 factor X에 영향을 받았을 때와,
        - X에 노출 된적 없었던 그룹 연구 대상 B가 factor X에 영향을 받지 않았을 때의 차이를 보게 됩니다.
        - 완전히 동일한 연구 대상에 2개의 다른 Treatment를 가할 수가 없습니다.
            - A -> Treatment -> A'
            - Note. we can not observe (A -> No Treatment -> A'') (시간이 지나면서 A도 Treatmnet가 아닌 다른 무언가에 영향을 받음)            
            - What we want to see: factor x => (A'-A'')

            - However, what we actually compare
            - B -> No Treatment -> B'(?) (시간이 지나면서 B도 뭔가 바뀜)
            - Then, we observe factor x => (A'-B') - (A-B)

    4. 9 page
        - Prediction은 Out of sample에서의 High Predictive Power를 목표로 합니다.
            - 이를 통해서 모든 Correlation을 고려하여 Bias를 줄이며,
            - 동시에 Overfitting 문제를 다루어 Variance를 최소화 합니다.

    5. 10 page
        - 따라서 Prediction 모델과, Causation 모델은 다른 두가지 목적으로 사용됩니다.
            - 어떠한 Factor X가 정말로 Outcome Y에 영향을 미쳤는지를 Causual하게 알고 싶으면,
            - Causation 모형을 사용하여야 하고
            - 예측률이 높은 모형을 만들고 싶으면 Prediction 모형을 쓰면 됩니다.

#### Question3. unbiased estimate란?
    
    unbiased estimator란 모델 추정량의 평균이 실제 모수값과 일치할 때를 뜻합니다.

        An estimator is said to be unbiased if its bias is equal to zero for all values of parameter θ. (https://en.wikipedia.org/wiki/Bias_of_an_estimator)
    
    아래의 자료의 그림 "Graphical illustration of bias and variance" 을 보시면, Low Bias에 대한 이해가 분명해 지실 것 같습니다.
    
![bias_vs_variance](bias_vs_variance.png)

    (http://scott.fortmann-roe.com/docs/BiasVariance.html)


## Module1.2 Tensorflow Keras API on Google Colab
#### Question1. 모델 만드는 법에 대해서 더 배우고 싶어요. 있는 코드 따라하는거야 문제 없는데 새로운 모델을 만들라고 하면 코드를 전혀 생각해 낼수 없을것 같습니다. (Work-in-Progress)

#### Question2. 은닉 레이어의 출력 값이 정해지는 기준( (tf.keras.layers.Dense(128, activation=tf.nn.relu) ) 시에 128이 정해지는 기준  ,  영상에서 레이어 한개는 뉴럴 네트워크 두개이상 딥 뉴럴 네트워크라고 설명을 했는데 같은값으로 은닉 레이어를 두개 만들었는데( model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) ) 이렇게 할 시 한 개와 어떤 차이점이 있는지. (Work-in-Progress)


#### Question3. Overfitting과 레이어 갯수의 결정 사이에 대한 연관성 (Work-in-Progress)
