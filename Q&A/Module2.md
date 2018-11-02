# Module1
## Module12.1 Convolutional Neural Networks
#### Question1. 이미지 데이터를 왜 Reshape 해주는 건가요? CNN에서 Filter와 Image에 대한 파라미터는 어떻게 설정해야 하나요?

    이미지 데이터의 regular한 모양은 (Number of image, Height of image, widgth of image, color channel (or output channel))입니다. 

    따라서 reshape(-1, IMG_SIZE =100, IMG_SIZE =100, 1) 을 통해, 
    모든 이미지에 대해서 (-1) (number), IMG_SIZE (width), IMG_SIZE (height), 1 (Grey Single Color)의 모양으로 맞추어 주었습니다.

    그리고, Conv2D(256, (3, 3), input_shape=X.shape[1:])) 에서 보시면 첫번쨰 256은 Number of Output Channel인데 Filter의 개수로 해석하시면 됩니다. 
    
    두번째, (3,3)는 width가 3 height가 3인 kernel에 대한 설정이며,
    input shape 부분에서는 X.shape[1:] 을 취하게되면, 이전에 모양이 (N, H, W, C) (N: number of image, H: height, W: width, C: color)로 맞춰어져 있었기 때문에
    X.shape[1:] 는 (H,W,C)의 모양을 넘겨 주게 됩니다~

    이렇게 하는 이유는 Batch Size 혹은 Number of Sample의 개수는 파라미터로 넘겨주지 않아도, 
    학습 과정에서 알게되는 정보이고, 나머지 H W C의 모양은 Conv2d에서 요구하는 image의 regular한 모양이기 때문입니다.

    추가적으로, Conv2D(256, (3, 3), input_shape=X.shape[1:])) 를 취하면 그 결과 Output shape를 model.summary()로 보았을때 (None, 98, 98, 256) 되는걸 보실수 있는데
    여기서 256이 사실은 이전에 Filter의 개수 이자 Color Channel이있던 자리 인걸 보실 수 있습니다.
    각각의 Filter를 통과해 나온 값들이 하나의 Output Channel(Color Channel)로 사용된다고 보시면 됩니다. 
    
#### Question2. CNN에서 Filters 갯수 256와 kernel_size 3x3는 어떻게 정하나요?

    좋은 질문이네요! :slightly_smiling_face: CNN에서 Design Choice를 어떻게 할지는, 
    굉장히 어려운 문제인데, 한 층에 대해서 Best Kernel Size를 정하기 보다는, 
    전체 네트워크를 고려했을 때 각 층마다 다양한 Kernel Size를 정하는 편이 실증적으로는 최적인 것으로 알고 있습니다. 

    또한, Kernel Size에 따라서 추상화 할수 있는 범위의 크기가 달라지는데, 
    Kernel의 개수와 함께 고려해 볼 점이 있습니다. Size가 작더라도 개수가 많으면 보다 큰 Size의 Kernel이 만들어내는 추상화 범위를 가질 수 있습니다.

    또한 주어진 문제마다, 다른 Design이 효과적일 수 있고, 일반적으로는 성능을 높히기 위해서, 가장 널리 쓰이는 Design을 실험한 이후에 해당 Accuracy가 원하는 Level인지 보고, 
    그렇지 않을 경우에 Parameter를 바꿔가면 다양하게 실험해 보는것 같습니다. 아래의 다양한 Parameter에 따른 Performance 차이를 참조해 보시면 좋을 것 같습니다. (https://github.com/ducha-aiki/caffenet-benchmark) 
    CNN에 관련된 논문에서 사용된 Parameter를 참조 바랍니다. (https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/)

    이 외에도 하드웨어에 따라서 학습 속도면에서 커널 사이즈를 어떻게 선택할지도 관련이 있는 것으로 알고 있습니다. 
    따라서 Universal한 design이 있다기 보다는, 주어진 상황과 리소스에 맞춰서 선택해야하는 문제가 있는 것 같습니다.

#### Question3. Cat 사진중에 140.jpg 파일은 아래 불러오는 과정중에서 Exception으로 처리가 됩니다. 해당 이미지는 왜 안불러지는건가요?

## Module12.2 Tensorboard on Google Colab

## Module12.3 Save and Load on Google Colab
#### Question1. 모델이 저장되는 방식이 h5 인가요? ProtoBuffer 포맷 즉 pb파일로 만드는법을 알려주세요

    해당 기능은 keras 자체적으로는 제공하지 않지만 tensorflow의 기능을 이용해서 pb 파일을 만드실 수 있습니다.

    아래의 코드에서 pb 파일 변환하는 법을 작성하였습니다.
    https://colab.research.google.com/drive/1Wts31gnRnForkl9QMn9oauVpKUzbhb6m

    자세한 튜토리얼은 아래의 링크를 참조 바랍니다.

    https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
    https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

## Module12.4 Recurrent Neural Networks

## Module12.5 Preprocessing for Cryptocurrency Data
   