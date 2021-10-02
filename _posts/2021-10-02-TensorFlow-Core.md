---
title:  "텐서 플로우(TensorFlow)"
excerpt: "텐서 플로우 내용 정리(https://www.tensorflow.org/)"

categories:
 - AI
tags:
 - [python, machine learning]

toc: true
toc_sticky: true

date: 2021-10-02
last_modified_at: 2021-10-02
---

## **0. 텐서 소개**

```python
#1 TensorFlow import
import tensorflow as tf
import numpy as np
```

지원되는 모든 `dtypes`은 `tf.dtypes.DType`에서 볼 수 있다.

텐서는 일종의 `np.arrays`와 같다.

```python
#2 TensorFlow DType Check
tf.dtypes.DType
```

모든 텐서는 `Python` 숫자 및 문자열과 같이 변경할 수 없다.

텐서의 내용은 업데이트할 수 없고, 새로운 텐서를 생성할 수만 있다.

---

## **1. 기초**

기본 텐서를 알아보자.

 **`"스칼라(Scalar)"` or `"순위-0(rank-0)"` Tensor**

 * 스칼라는 단일 값을 포함하며 "축"은 없다.

```python
#3 기본적으로 int32 텐서가 된다. 아래 유형 참조
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
```

```
tf.Tensor(4, shape=(), dtype=int32)
```

**`"벡터(Vector)"` or `"순위-1(rank-1)"` Tensor**

* 벡터 텐서는 값의 목록과 같다.
* 벡터에는 하나의 축이 있다.

```python
#4 벡터를 이용해 float 텐서를 만들어 보자.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
```

```
tf.Tensor([2. 3. 4.], shape=(3,), dtype=float32)
```

**`"행렬(matrix)"` or `"순위-2(rank-2)"` Tensor**
* 행렬에는 두 개의 축이 있다.

```python
#5 구체적으로 설명하려면 생성 시 dtype을 설정할 수 있다.
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)
```

텐서에는 더 많은 축이 있을 수 있고 여기선 세 개의 축이 있는 텐서가 사용된다.
