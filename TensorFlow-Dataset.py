import tensorflow_datasets as tfds

# MNIST 데이터셋 로드
mnist_dataset, info = tfds.load(name="mnist", split="train", with_info=True)

# 데이터 확인
for example in mnist_dataset.take(1):
    image, label = example["image"], example["label"]
    # 이미지 및 레이블 출력 또는 다른 작업 수행