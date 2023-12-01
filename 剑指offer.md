### 剑指 Offer 09. 用两个栈实现队列

> 用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

```

输入：

["CQueue","appendTail","deleteHead","deleteHead","deleteHead"]

[[],[3],[],[],[]]

输出：[null,null,3,-1,-1]

```

```
输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]

```
```python
class CQueue:

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def appendTail(self, value: int) -> None:
        self.stack1.append(value)


    def deleteHead(self) -> int:
        if self.stack2:
            return self.stack2.pop()
        if not self.stack1:
            return -1
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        return self.stack2.pop()


# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
```
> 思考：可以利用两个栈完成队列的构建，通过一个栈暂时存储入栈元素，另一个栈调整元素顺序来实现FIFL

### 剑指 Offer 30. 包含 min 函数的栈
> 定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。
> 
```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
```

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.minStack = []
    def push(self, node):
        # write code here
        self.stack.append(node)
        if self.minStack == []:
            self.minStack.append(node)
        else:
            min = self.minStack[len(self.minStack) - 1]
            if node <= min:
                self.minStack.append(node)
            else:
                self.minStack.append(min)
    def pop(self):
        self.stack.pop()
        self.minStack.pop()
    def top(self):
        # write code here
        return self.stack[len(self.stack) - 1]
    def min(self):
        # write code here
        return self.minStack[len(self.minStack) - 1]



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.min()
```
> 思路：使用辅助栈来完成最小值的筛选。在将元素入栈时保证辅助栈栈顶为当前最小元素

### 单链表的复制

```python

#  这里是单链表的定义
class Node(object):
    def __init__(self, item):
        self.val = item
        self.next = None
 
 
def copyList(head):
    cur = head # 当前节点
    pre = Node(0) # 复制新链表的起点
    dum = pre # 我们用dum来保存新链表的首个节点
    while cur:
        node = Node(cur.val) # 复制链表的第一个节点
        pre.next = node # 让新链表的起点指向 第一个节点
        cur = cur.next # 旧链表下一个节点
        pre = node # 新链表下一个节点
    return dum.next # 返回新链表的起点， 不用pre因为pre已经到了链表的最后


```

### 剑指 Offer 06. 从尾到头打印链表

> 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
```
输入：head = [1,3,2]
输出：[2,3,1]
```
> 普通方法：一个栈实现倒排，然后列表翻转
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        stack = []
        while head:
            stack.append(head.val)
            head = head.next
        return stack[::-1]
```

> 递归法

```python
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        # 递归的终止条件
        if not head :
            return []
        else:
            return self.reversePrint(head.next) + [head.val]
```

> 思路：
>
> 递推阶段： 每次传入 head.next ，以 head == None（即走过链表尾部节点）为递归终止条件，此时返回空列表 [] 。
> 回溯阶段： 利用 Python 语言特性，递归回溯时每次返回 当前 list + 当前节点值 [head.val] ，即可实现节点的倒序输出。

### 剑指 Offer 24. 反转链表

> 定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

```
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

> 双指针法

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre, cur = None, head
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre

```

> 思考：利用一个暂存节点来保存pre.next
> 在遍历链表时，将当前节点的 next指针改为指向前一个节点。由于节点没有引用其前一个节点，因此必须事先存储其前一个节点。在更改引用之前，还需要存储后一个节点。最后返回新的头引用。

### 剑指 Offer 05. 替换空格

> 请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。
```
输入：s = "We are happy."
输出："We%20are%20happy."
```
```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        res = []
        for c in s:
            if c == ' ': res.append("%20")
            else: res.append(c)
        return "".join(res)
```
> 思考：注意join的使用，怎么将list转化为string的

### 剑指 Offer 58 - II. 左旋转字符串
>  字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

```
输入: s = "abcdefg", k = 2
输出: "cdefgab"
输入: s = "lrloseumgh", k = 6
输出: "umghlrlose"
```
```python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        return s[n:] + s[:n]
```



### 剑指 Offer 53 - I. 在排序数组中查找数字 I
> 统计一个数字在排序数组中出现的次数。
```
输入: nums = [5,7,7,8,8,10], target = 8
输出: 2

输入: nums = [5,7,7,8,8,10], target = 6
输出: 0
```


```python
class Solution:
    def search(self, nums: [int], target: int) -> int:
        # 搜索右边界 right
        i, j = 0, len(nums) - 1
        while i <= j:
            m = (i + j) // 2
            if nums[m] <= target: i = m + 1
            else: j = m - 1
        right = i
        # 若数组中无 target ，则提前返回
        if j >= 0 and nums[j] != target: return 0
        # 搜索左边界 left
        i = 0
        while i <= j:
            m = (i + j) // 2
            if nums[m] < target: i = m + 1
            else: j = m - 1
        left = j
        return right - left - 1

```

> 思考：对于有序列表的查找，可以考虑使用二分查找而不是暴力遍历，可以将时间复杂度降至log(N)





### 剑指 Offer 03. 数组中重复的数字

> 找出数组中重复的数字。

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

```
输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3
```
> 哈希表
```python
class Solution:
    def findRepeatNumber(self, nums: [int]) -> int:
        dic = set()
        for num in nums:
            if num in dic: return num
            dic.add(num)
        return -1
```
> 思考：可以利用集合的特性来返回重复元素

> 原地交换
```python
class Solution:
    def findRepeatNumber(self, nums: [int]) -> int:
        i = 0
        while i < len(nums):
            if nums[i] == i:
                i += 1
                continue
            if nums[nums[i]] == nums[i]: return nums[i]
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        return -1

```
> 利用key和value是一对多的关系来查找同一个索引的不同值





### 剑指 Offer 04. 二维数组中的查找

> 在一个 n * m 的二维数组中，每一行都按照从左到右 非递减 的顺序排序，每一列都按照从上到下 非递减  的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

```
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```
给定 target = 5，返回 true。

给定 target = 20，返回 false。

```python
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        i, j = len(matrix) - 1, 0
        while i >= 0 and j < len(matrix[0]):
            if matrix[i][j] > target: i -= 1
            elif matrix[i][j] < target: j += 1
            else: return True
        return False
```
> 可以利用矩阵非递减的特点将其转化为二叉搜索树，从而只需要O(M+N)的时间复杂度


### 剑指 Offer 11. 旋转数组的最小数字
>把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
>给你一个可能存在 重复 元素值的数组 numbers ，它原来是一个升序排列的数组，并按上述情形进行了一次旋转。请返回旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一次旋转，该数组的最小值为 1。  
>注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。


```
示例 1：

输入：numbers = [3,4,5,1,2]
输出：1
示例 2：

输入：numbers = [2,2,2,0,1]
输出：0
```

```python
class Solution:
    def minArray(self, numbers: [int]) -> int:
        i, j = 0, len(numbers) - 1
        while i < j:
            m = (i + j) // 2
            if numbers[m] > numbers[j]: i = m + 1
            elif numbers[m] < numbers[j]: j = m
            else: j -= 1
        return numbers[i]
```

> 对于有序数组的查找问题，优先考虑使用二分查找来降低时间复杂度





### 剑指 Offer 32 - I. 从上到下打印二叉树
> 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

 

例如:
给定二叉树: [3,9,20,null,null,15,7],
```
    3
   / \
  9  20
    /  \
   15   7
```
返回：
```
[3,9,20,15,7]
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root: return []
        res, queue = [], collections.deque()
        queue.append(root)
        while queue:
            node = queue.popleft()
            res.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        return res
```




> 思考：对于二叉树的广度优先遍历，可以考虑使用队列来完成。因为每一层的顺序都是从左到右的，可以先将左右子树分别入队，然后再对左右子树进行运算。



### 剑指 Offer 26. 树的子结构
输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
```
给定的树 A:

     3
    / \
   4   5
  / \
 1   2
给定的树 B：

   4 
  /
 1
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

示例 1：

输入：A = [1,2,3], B = [3,1]
输出：false
示例 2：

输入：A = [3,4,5,1,2], B = [4,1]
输出：true
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        def recur(A, B):
            if not B: return True
            if not A or A.val != B.val: return False
            return recur(A.left, B.left) and recur(A.right, B.right)

        return bool(A and B) and (recur(A, B) or self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B))
```
> 思考：对于数的查找问题，可以考虑用递归遍历。这里因为包含查找和匹配的过程，所以使用了两次递归

### 剑指 Offer 53 - II. 0～n-1 中缺失的数字
一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

```
示例 1:

输入: [0,1,3]
输出: 2
示例 2:

输入: [0,1,2,3,4,5,6,7,9]
输出: 8
```
使用类似二分查找的思想查找
```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        # for i in range(len(nums)):
        #     if nums[i]!=i:
        #         return i
        # return len(nums)
        i,j=0,len(nums)-1
        while i<=j:
            mid=(i+j)//2
            if nums[mid]==mid:
                 i=mid+1
            else: 
                j=mid-1
        return i
```
使用哈希集合来判断
```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        s = set(nums)
        for i in range(len(nums) + 1):
            if i not in s:
                return i

```
数学方法求解
```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums) + 1
        total = n * (n - 1) // 2
        arrSum = sum(nums)
        return total - arrSum

```


### 剑指 Offer 27. 二叉树的镜像
请完成一个函数，输入一个二叉树，该函数输出它的镜像。


```
例如输入：

     4
   /   \
  2     7
 / \   / \
1   3 6   9
镜像输出：

     4
   /   \
  7     2
 / \   / \
9   6 3   1

 

示例 1：

输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root: return
        temp = root.left
        root.left = self.mirrorTree(root.right)
        root.right = self.mirrorTree(temp)
        return root
```
> 可以使用递归来完成树的遍历，同时利用temp暂存节点实现交换

### 剑指 Offer 10- I. 斐波那契数列

写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：
```
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
```
斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。


```
示例 1：

输入：n = 2
输出：1
示例 2：

输入：n = 5
输出：5
```
```python

class Solution:
    def fib(self, n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, (a + b) % 1000000007
        return a
```
> 可以使用动态规划的方法来避免递归，从而降低时间复杂度。


```python
# 取余运算的细节
#大数/小数：因为得出的商和整除得出的一致，所以直接按照这个公式（余数=除数-被除数*商）即可。
print(9//7)  #1
print(9%7)   #2


#小数/大数：因为得出的商和整除得出的一致，所以直接按照这个公式（余数=除数-被除数*商）即可。
#这里也可以说：只要正数与正数是小数/大数 的，商都是0 ，所以余数是他本身。
print(7//9) #0
print(7%9)  #7

```


### 剑指 Offer 42. 连续子数组的最大和
输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

 
```
示例1:

输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
 

提示：

1 <= arr.length <= 10^5
-100 <= arr[i] <= 100
```


```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            if nums[i-1] > 0:
                nums[i] += nums[i-1]
            else:
                continue
        return max(nums)
```