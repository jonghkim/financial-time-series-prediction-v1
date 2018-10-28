# Module1
## Module1.1 Introduction
#### [주간 질문] AI를 통해 Finance의 어떤 문제들을 해결할 수 있을까요? 

    1. Increasing security: security와 관련된 빅데이터 분석을 통한 보안 향상. (Anti-Money Laundering, fraud detection 등등)
    2. Reducing processing times: financial documentation 처리에 걸리는 시간을 줄여주는 효과.
    3. Algorithmic trading:trading pattern을 찾고, trends를 예측하기 위하여 AI를 적용시켜서, 더 성공적인 stock trading decisions을 내리도록 함.
    4. Customer recommendation / Credit lending: consumer data, financial lending 혹은 insurance results등의 자료를 기반으로, 고객들의 관심분야를 추천에서부터, 대출 여부를 평가하는 등의 분야에도 적용 가능함.
    5. Porfolio Management

#### Question1. 머신러닝이 할수 있는 예측에 대하여 궁금합니다. 일부에서는 스포츠 승률이라던지 심지어는 복권당첨도 예측하던데요. (Work-in-Progress)


#### Question2. 과연 딥러닝으로 금융시장을 이길수있는 전략을 도출할수있을까요? (Work-in-Progress)


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

#### Question2. 6-10 Page 및 7,8 Page에 대한 추가 설명 (Work-in-Progress)

#### Question3. unbiased estimate란? (Work-in-Progress)

## Module1.2 Tensorflow Keras API on Google Colab
#### Question1. 모델 만드는 법에 대해서 더 배우고 싶어요. 있는 코드 따라하는거야 문제 없는데 새로운 모델을 만들라고 하면 코드를 전혀 생각해 낼수 없을것 같습니다. (Work-in-Progress)

#### Question2. 은닉 레이어의 출력 값이 정해지는 기준( (tf.keras.layers.Dense(128, activation=tf.nn.relu) ) 시에 128이 정해지는 기준  ,  영상에서 레이어 한개는 뉴럴 네트워크 두개이상 딥 뉴럴 네트워크라고 설명을 했는데 같은값으로 은닉 레이어를 두개 만들었는데( model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) ) 이렇게 할 시 한 개와 어떤 차이점이 있는지. (Work-in-Progress)


#### Question3. Overfitting과 레이어 갯수의 결정 사이에 대한 연관성 (Work-in-Progress)
