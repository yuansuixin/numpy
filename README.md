

- 机器学习相关的知识点和代码，我都是使用的jupyter-notebook，大家可以是用jupyter打开，推荐使用anaconda。直接下载安装就可以，方便快捷。




## numpy库

###### 主要处理高维度的数据
import numpy as np
## 一、创建ndarray
#### 1. 使用np.array()由Python list创建
注意：
- numpy默认ndarray的所有元素的类型是相同的
- 如果传进来的列表中包含不同的类型，则统一为同一类型，优先级：str>float>int

```
arr1 = np.array([1,2,'3.0',4,6,7,9])
display(arr1)

array(['1', '2', '3.0', '4', '6', '7', '9'], dtype='<U11')
```
#### 2.使用np的routines函数创建
- np.ones(shape, dtype=None, order='C') 参数shape表示几行几列
```
 np.ones(shape=2,dtype=np.int)
array([1, 1])

 np.ones(shape=(2,3))
array([[1., 1., 1.],
       [1., 1., 1.]])
```
- np.zeros(shape, dtype=float, order='C')
```
 np.zeros(shape=(3,3),dtype=np.int)
array([[0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]])
```

- np.full(shape, fill_value, dtype=None, order='C')

```
np.full(shape=(2,3),fill_value=6)

array([[6, 6, 6],
       [6, 6, 6]])
```
-  np.eye(N, M=None, k=0, dtype=float)
  对角线为1其他的位置为0

```
#单位矩阵
np.eye(N=4,M=4,k=0)

array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])
```
- np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```
# 指定元素个数来生成一个等差数列
np.linspace(start=0,stop=100,num=10,endpoint=False)
array([ 0., 10., 20., 30., 40., 50., 60., 70., 80., 90.])
```
-  np.arange([start, ]stop, [step, ]dtype=None)

```
#指定步长来创建一个等差数列
np.arange(0,100,step=5)

array([ 0,  5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
       85, 90, 95])
```
- np.random.randint(low, high=None, size=None, dtype='l')
```
np.random.randint(0,100,size=5)
np.random.randint(0,100,size=(2,3))
np.random.randint(0,100)
84
```

- np.random.randn(d0, d1, ..., dn)标准正太分布np.random.normal()

```
np.random.randn(2,3,3)

array([[[ 0.16287731, -1.81813488,  0.55260039],
        [ 0.61774996, -0.66366244,  0.28501785],
        [ 0.85960777, -0.55725215, -0.9356485 ]],

       [[-1.29262526,  0.46597457,  0.98603002],
        [ 0.42601417, -0.566575  ,  0.85771228],
        [ 0.46329296, -0.55985631, -0.39543827]]])
```
```
# 可以控制期望值和方差变化
# loc 期望值 默认0
# scale 方差 默认值1
np.random.normal(loc=10,scale=1,size=(2,3))

array([[10.73222936,  9.68440034, 10.55653069],
       [10.05839774, 12.32754481,  9.78746624]])
```
- np.random.random(size=None)生成0-1的随机数，左闭右开
```
data = np.random.random(size=(100,100,3))
np.random.random(size=30000).reshape((100,100,3))
这两种写法等价
```
## 二、ndarray的属性
4个必记参数： 
ndim：维度 
shape：形状（各维度的长度） 
size：总长度
dtype：元素类型

## 三、ndarray的基本操作
#### 1. 索引
一维与列表完全一致，多维则同理
```
arr2 = np.random.randint(0,100,size=(3,5))
arr2[0][0]
arr2[0,0]
以上两种写法完全等价
```
#### 2. 切片
一维与列表完全一致，多维则同理,切片操作不会改变原数据

```
arr2 = np.random.randint(0,100,size=(3,5))
#行切片
arr2[0:1]
array([[59, 66, 52, 96, 28]])
#列切片
arr2[:,0:2]
array([[59, 66],
       [12, 45],
       [73, 98]])
#需求，将二维数组arr2进行行列逆向变换处理
arr2[::-1,:]
array([[73, 98, 51, 58, 34],
       [12, 45, 99, 66, 16],
       [59, 66, 52, 96, 28]])
```
#### 3. 变形
使用reshape函数，注意参数是一个tuple！

#### 4.级联
1.np.concatenate()
级联需要注意的点：
- 级联的参数是列表：一定要加中括号或小括号
- 维度必须相同
- 形状相符
- 【重点】级联的方向默认是shape这个tuple的第一个值所代表的维度方向
- 可通过axis参数改变级联的方向

```
n1 = np.ones(shape=(6,3))
n2 = np.zeros(shape=(6,3))
display(n1,n2)
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])
#axis就表示维度的方向,水平，垂直
np.concatenate((n1,n2),axis=1)
array([[1., 1., 1., 0., 0., 0.],
       [1., 1., 1., 0., 0., 0.],
       [1., 1., 1., 0., 0., 0.],
       [1., 1., 1., 0., 0., 0.],
       [1., 1., 1., 0., 0., 0.],
       [1., 1., 1., 0., 0., 0.]])
```
2.np.hstack与np.vstack  
水平级联与垂直级联,处理自己，进行维度的变更
h表示水平，v表示垂直

#### 5.切分
与级联类似，三个函数完成切分工作：
- np.split
- np.vsplit
- np.hsplit

```
arr = np.random.randint(0,100,size=(6,6))
array([[ 0, 89, 23, 35, 43, 86],
       [53, 74, 60, 42, 78, 46],
       [23, 78, 47, 67,  5, 76],
       [25, 41, 20, 84, 23, 52],
       [17, 31, 30,  5, 31,  3],
       [39, 91,  4, 85, 14, 62]])
#指定切分的块数来进行切分，axis控制轴向
np.split(arr,indices_or_sections=3,axis=1)
[array([[ 0, 89],
        [53, 74],
        [23, 78],
        [25, 41],
        [17, 31],
        [39, 91]]), 
array([[23, 35],
        [60, 42],
        [47, 67],
        [20, 84],
        [30,  5],
        [ 4, 85]]), 
array([[43, 86],
        [78, 46],
        [ 5, 76],
        [23, 52],
        [31,  3],
        [14, 62]])]
#自定义切分
#n m  切分范围为 0~n ,n~m, m~last
np.split(arr,indices_or_sections=[3,5],axis=1)
[array([[ 0, 89, 23],
        [53, 74, 60],
        [23, 78, 47],
        [25, 41, 20],
        [17, 31, 30],
        [39, 91,  4]]), 
    array([[35, 43],
        [42, 78],
        [67,  5],
        [84, 23],
        [ 5, 31],
        [85, 14]]), 
    array([[86],
        [46],
        [76],
        [52],
        [ 3],
        [62]])]

```
#### 6.副本
所有赋值运算不会为ndarray的任何元素创建副本。对赋值后的对象的操作也对原来的对象生效。

可使用copy()函数创建副本


## 四、ndarray的聚合操作
#### 1.求和np.sum

```
n1 = np.arange(101)
np.sum(n1)
5050
n2 = np.random.randint(0,10,size=(3,3))
array([[8, 8, 5],
       [6, 0, 0],
       [8, 2, 6]])
np.sum(n2,axis=None)#所有值求和
43
np.max(n2,axis=1)#每一行的最大值
array([8, 6, 8])
#如果操作的是一维数组，没有问题，
#如果是二维数组，要指定axis，不然得到的索引无法使用，需要对原数组做扁平化处理
np.argmin(n2,axis=1)#每一行的最小值
array([2, 1, 1], dtype=int32)
```
```
n1 = np.array([1,False,3,True,5,6])
array([1, 0, 3, 1, 5, 6])
np.all(n1)
False
```
None和np.nan类型
- None是一个对象类型
- np.nan只是一个bool类型的值，效率差别很大
#### 2. 最大值和最小值np.max,np.min
any()  有True返回True
all()  有False返回False
- 任何值与np.nan计算结果都为nan

```
# 处理包含空值的集合的聚合
np.nansum(n2)
10.0
```
#### 3.其他聚合操作
```
Function Name    NaN-safe Version    Description
np.sum    np.nansum    Compute sum of elements
np.prod    np.nanprod    Compute product of elements
np.mean    np.nanmean    Compute mean of elements
np.std    np.nanstd    Compute standard deviation
np.var    np.nanvar    Compute variance
np.min    np.nanmin    Find minimum value
np.max    np.nanmax    Find maximum value
np.argmin    np.nanargmin    Find index of minimum value
np.argmax    np.nanargmax    Find index of maximum value
np.median    np.nanmedian    Compute median of elements
np.percentile    np.nanpercentile    Compute rank-based statistics of elements
np.any    N/A    Evaluate whether any elements are true
np.all    N/A    Evaluate whether all elements are true
np.power 幂运算
```
np.sum 和 np.nansum 的区别 nan not a number
- 操作文件
  使用pandas打开文件president_heights.csv 获取文件中的数据

## 五、 ndarray的矩阵操作
#### 1. 基本操作
- 算术运算
    - 加减都是对应位置相互做加减的算术运算
    - 注意，* 只是两个矩阵对应位置的数据做的算术的相乘，并不是积

```
n1 = np.random.randint(0,10,size=(3,3))
n2 = np.random.randint(0,10,size=(3,3))
display(n1,n2)
array([[3, 8, 0],
       [7, 9, 7],
       [3, 4, 6]])
array([[5, 8, 4],
       [9, 8, 2],
       [7, 8, 1]])
n1+n2
array([[ 8, 16,  4],
       [16, 17,  9],
       [10, 12,  7]])
n1*n2
array([[15, 64,  0],
       [63, 72, 14],
       [21, 32,  6]])
```
- 矩阵积
  ```np.dot(n1,n2)```

#### 2.广播机制
【重要】ndarray广播机制的两条规则
- 规则一：为缺失的维度补1
- 规则二：假定缺失元素用已有值填充


例1：
m = np.ones((2, 3))
a = np.arange(3)
求M+a

```
m = np.ones((2,3))
a = np.array([5])
display(m,a)
array([[1., 1., 1.],
       [1., 1., 1.]])
array([5])
m+a
array([[6., 6., 6.],
       [6., 6., 6.]])
```
## 六、ndarray的排序
### 1. 快速排序
np.sort()与ndarray.sort()都可以，但有区别：
- np.sort()不改变输入
- ndarray.sort()本地处理，不占用空间，但改变输入

```
arr = np.random.randint(0,100,size=10)
array([90, 39, 30,  1, 81, 65,  6, 20, 79, 25])
sort_arr = np.sort(arr)
sort_arr = array([ 1,  6, 20, 25, 30, 39, 65, 79, 81, 90])
arr.sort()
arr = array([ 1,  6, 20, 25, 30, 39, 65, 79, 81, 90])
```
### 2. 部分排序
np.partition(a,k)

有的时候我们不是对全部数据感兴趣，我们可能只对最小或最大的一部分感兴趣。
- 当k为正时，我们想要得到最小的k个数
- 当k为负时，我们想要得到最大的k个数

```
np.partition(arr,-2)
array([79, 25, 30,  1, 39, 65,  6, 20, 81, 90])
```



#### 补充
- 求数组中的最大值的索引使用argmax()函数

- 根据第3列来对一个5*5矩阵排序
![Untitled-1-201845222257](http://p693ase25.bkt.clouddn.com/Untitled-1-201845222257.png)

- sum(a) 求数组每一列的和
- a.sum()求数组所有元素的和

```
>>>a = np.array([[1, 2], [3, 4]])  
>>>np.mean(a) # 将上面二维矩阵的每个元素相加除以元素个数（求平均数）  
2.5  
>>>np.mean(a, axis=0) # axis=0，计算每一列的均值  
array([ 2.,  3.])  
>>>np.mean(a, axis=1) # 计算每一行的均值  
array([ 1.5,  3.5])  
```
