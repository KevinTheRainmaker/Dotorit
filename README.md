# Dotorit

Price Suggestion Model for Second-hand items

<강빈이형 보세요>

주요 notebook 파일은 DOTORIT/src/model 에 있는 base_model.ipynb 과 Item_extractor.ipynb 파일입니다.

1. base_model.ipynb
   base_model 파일은 Feature Vector 를 뽑아내기 위해 학습한 모델이며, ImageNet 으로 기존에 학습되어 있는 VGG-16 모델에, 저희가 갖고 있는 데이터들을 전이 학습시켰습니다.

- (전이학습 할 때, Image Classifier 모델로 학습 시켜서, 마지막 Layer output 이 (None, category 개수) 로 구성되어 있습니다. 이 Layer 의 Output 을 Feature 로 사용했습니다.

2. Item_extractor.ipynb
   base_model 의 마지막 Layer 의 Output 을 Feature 로 사용했으며, Image 파일 이름과 각 Image 의 Feature Vector 를 미리 추출하여, DOTORIT/src/saved_data 에 저장해놓았습니다.

- _따라서, 모델을 건드리지 않고 유사도가 높은 아이템을 추출하는 것을 확인하고 싶으시다면, Item_extractor 노트북 파일만 실행해도 될 것입니다._

덤으로.. 성능이 최악입니다. 데이터 부족 문제일 수도, 정규화 과정에서 문제가 있었을 수도 있을 것 같습니다. 이 부분은 이번 주 내에 한번 만져보면서, 확인해보겠습니다.

성능이 충격적이여서 우선은 속도 부분에서 이점을 얻을 수 있는 FAISS 라이브러리는 적용하지 않았습니다. FAISS 라이브러리 적용해서 시간 단축하는 코드는 금방 짤 수 있을 것 같아서, 결과가 조금 개선이 되면 코드 수정하겠습니다.

좋은 명절 되셨길 바랍니다.

## <양현준 보시오>

img_search_with_faiss.py로 끝냈시오

임베딩에 사용한 모델은 경량 모델인 MobieNetV2이고 최종 벡터 사이즈는 1001 차원

이후 FAISS를 이용하여 train data를 모두 벡터화하고 index에 더해서 저장했음

한번 저장 후엔 바로 불러올 수 있도록 index 파일이 있으면 벡터화 과정 없이 load 하게 만들었고, 눈으로 볼 수 있게 csv도 추가함

이후 supervision으로 가장 근접한 k개의 이미지 출력하도록 함

아래는 결과

### Query Image

![shoes_example](https://github.com/hy30n80/Dotorit/assets/76294398/7b269742-71a3-4676-a889-dc71712b89c8)

### Top-2 Image from Train set (actually not a train)

![shoes_13](https://github.com/hy30n80/Dotorit/assets/76294398/dc2fc31a-2fe2-4095-93c0-0e90d1839330)

![shoes_15](https://github.com/hy30n80/Dotorit/assets/76294398/5411a39f-bbd8-4f76-ace4-694c790ffa60)
