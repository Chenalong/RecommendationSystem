__author__ = 'dragon'


from __future__ import division
import numpy as np
import scipy as sp

class  Item_based_C:
    def __init__(self,X):
        self.X=np.array(X)                                  #把X转换成numpy中的数组
        print "the input data size is ",self.X.shape
        self.movie_user={}                                  #用户对某个电影的评分字典
        self.user_movie={}                                  #电影在某个用户下的评分字典
        self.ave=np.mean(self.X[:,2])                       #所有用户评分的均值
        for i in range(self.X.shape[0]):
            uid=self.X[i][0]
            mid=self.X[i][1]
            rat=self.X[i][2]
            self.movie_user.setdefault(mid,{})              #setdefault函数，如果字典中又mid，则返回对应的value，如果没有，在赋值操作
            self.user_movie.setdefault(uid,{})
            self.movie_user[mid][uid]=rat
            self.user_movie[uid][mid]=rat
            self.similarity={}
        pass

    def sim_cal(self,m1,m2):                                # 求解m1 和m2 电影的相识度
        self.similarity.setdefault(m1,{})
        self.similarity.setdefault(m2,{})
        self.movie_user.setdefault(m1,{})
        self.movie_user.setdefault(m2,{})
        self.similarity[m1].setdefault(m2,-1)
        self.similarity[m2].setdefault(m1,-1)

        if self.similarity[m1][m2]!=-1:
            return self.similarity[m1][m2]
        si={}                                               #存储对m1和m2都有评分的用户
        for user in self.movie_user[m1]:
            if user in self.movie_user[m2]:                 #判断时间复杂度是O(n)  可以用另一种写法是O(logn)
                si[user]=1
        n=len(si)
        # 如果没有相同用户设置为0合适吗？？？？？  0 表示不线性相关（可能曲线相关）1 表示正相关 -1 表示负相关
        if (n==0):
            self.similarity[m1][m2]=1
            self.similarity[m2][m1]=1
            return 1

        #简单相关系数又称皮尔逊相关系数来计算相关度
        s1=np.array([self.movie_user[m1][u] for u in si])
        s2=np.array([self.movie_user[m2][u] for u in si])
        sum1=np.sum(s1)
        sum2=np.sum(s2)
        sum1Sq=np.sum(s1**2)
        sum2Sq=np.sum(s2**2)
        pSum=np.sum(s1*s2)
        num=pSum-(sum1*sum2/n)
        den=np.sqrt((sum1Sq-sum1**2/n)*(sum2Sq-sum2**2/n))
        if den==0:
            self.similarity[m1][m2]=0
            self.similarity[m2][m1]=0
            return 0
        self.similarity[m1][m2]=num/den
        self.similarity[m2][m1]=num/den
        return num/den

    #预测用户对某个电影的打分
    def pred(self,uid,mid):
        sim_accumulate=0.0
        rat_acc=0.0
        for item in self.user_movie[uid]:
            sim=self.sim_cal(item,mid)
            if sim<0:continue
            #print sim,self.user_movie[uid][item],sim*self.user_movie[uid][item]
            rat_acc+=sim*self.user_movie[uid][item]
            sim_accumulate+=sim
        #print rat_acc,sim_accumulate
        if sim_accumulate==0: #no same user rated,return average rates of the data
            return  self.ave
        return rat_acc/sim_accumulate

    def test(self,test_X):
        test_X=np.array(test_X)
        output=[]
        sums=0
        print "the test data size is ",test_X.shape
        for i in range(test_X.shape[0]):
            pre=self.pred(test_X[i][0],test_X[i][1])
            output.append(pre)
            #print pre,test_X[i][2]
            sums+=(pre-test_X[i][2])**2
        rmse=np.sqrt(sums/test_X.shape[0])
        print "the rmse on test data is ",rmse
        return output