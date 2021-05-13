
import requests
import time
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.svm import SVR
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

import seaborn as sns

def height(x,y):
#the height function
    return x**2+y**2

def get_data():
    
    #利用爬虫技术从微博上爬微博热搜

    url='https://s.weibo.com/top/summary'
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.1;Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
    cookies={'Cookie':'_s_tentry=www.baidu.com; UOR=www.baidu.com,s.weibo.com,www.baidu.com; Apache=2037189309823.093.1606295580529; SINAGLOBAL=2037189309823.093.1606295580529; ULV=1606295580538:1:1:1:2037189309823.093.1606295580529:'}
    r=requests.get(url,headers=headers,cookies=cookies)
    if(r.status_code==200):
        print(len(r.text))
        #f=open('text.txt','w+')
        #f.write(r.text)

        
        #从网页源码中提取数据
        
        soup=BeautifulSoup(r.text,'html.parser')
        datalist=soup.find_all('td','td-02')
        datadic={}
        labels=[]
        heat=[]
        bar_width=2
        ticks=[]
        for i in range(len(datalist)):
            if(i==0):
                continue
            else:
                datadic[datalist[i].a.string]=datalist[i].span.string
                labels.insert(0,datalist[i].a.string)
                heat.insert(0,float(datalist[i].span.string))
                ticks.append(float(datalist[i].span.string))
                
            if(i>25):
                break
        return heat,labels,ticks

    
def make_dic(labels,heat):
    dic={}
    for i in range(len(labels)):
        dic[labels[i]]=heat[i]
    print('生成字典')
    return dic
    
def insert(dataframe,heat,labels,ticks):
    print('insert中的df结构:{}'.format(dataframe.shape))
    
    '''
    print('热搜排名:')
    for i in labels:
        print('{}'.format(i))    
    #print(dataframe.columns)
    '''
    print('')
    try:
        if(len(dataframe.columns)==0):
            array_heat=np.array(heat)
            array_heat=array_heat[np.newaxis,:]
            dataframe=pd.DataFrame(array_heat,columns=labels)
            
        else:
           
            dic=make_dic(labels,heat)
            dataframe=dataframe.append(dic,ignore_index=True)
            
            
            
            '''
            new=np.setdiff1d(labels,dataframe.columns)
            
            for i in new:
                ai=labels.index(i)
                print('新热点：{},排名:{}'.format(i,ai))
                
                print(heat[ai])
            print('')
            '''
            
    except Exception as e:
        print('发生异常:{}'.format(e))
    
    return dataframe

def make_dataSet(df_heat):
    #对爬去的进行数据空值，异常值清洗处理
    #pct_one_heat=df_heat.pct_change()
    dataSet=pd.DataFrame()
    '''
    one_df_heat_mean=df_heat.rolling(window=2).mean()
    five_df_heat_mean=df_heat.rolling(window=5).mean()
    seven_df_heat_mean=df_heat.rolling(window=7).mean()
    
    
    
    task=[one_df_heat_mean,five_df_heat_mean,seven_df_heat_mean]
    for i_name in task:
        
        
        if(i_name.empty):
            return False
        else:
            i_name.dropna(inplace=True)
            
            
           
            dic=make_dic(i_name.columns,i_name[-1:].values[0])
            dataSet=dataSet.append(dic,ignore_index=True)
            #print(dic)
            
                
    print()
    '''
    
    dataSet=df_heat.pct_change(periods=1)
    dataSet.fillna(0,inplace=True)
    dataSet[np.isinf(dataSet)]=1
    dataSet.to_csv('pct.csv',encoding='utf_8_sig')
    
    return dataSet




def task_dropna(task):
    for i in task:
        i.dropna(inplace=True)
    
def gen_train_set():
    #生成供人工神经网络预测的数据训练集
    a=pd.read_csv('wb_heat_120.csv',encoding='utf_8_sig',index_col=0)
    '''
    b=pd.read_csv('two_mean.csv',encoding='utf_8_sig',index_col=0)
    c=pd.read_csv('five_mean.csv',encoding='utf_8_sig',index_col=0)
    d=pd.read_csv('seven_mean.csv',encoding='utf_8_sig',index_col=0)
    task_dropna([a,b,c,d])
    '''
    train_set=np.zeros(shape=(1,4))
    dataSet=make_dataSet(a)
    test_labels=['戚薇雾莲仙女裙','毛衣如何穿出时髦感','金在中身材', '名侦探学院', '英国议员夺走女王权杖抗议', '黄多多短发' ,'微博之夜投票', '黄子韬怒斥偷拍者' ,'上万只乌鸦遮天蔽日侵袭加拿大' ,'义乌市图书馆停用制热空调', 'BLACKPINK演唱会延期', '一只小动物从嫦五返回器前跑过', '巴西将使用中国新冠疫苗', '广东药科大学失联男生已离世' ,'39岁全智贤状态', '王自健前妻发文', '劳斯莱斯回应拼多多122万元补贴','中国盲盒出口增速超400%', '川航确诊飞行员在江油密接均为阴性', '狗仔镜头下的关晓彤','听起来像日文名的中文名', '邓超对孙俪的字佩服得五体投地' ,'义乌限电晚上关闭所有路灯' ,'还以为进了青青草原' ,'男子扯掉并脚踢国旗被依法行拘' ,'王一博有翡老年妆' ,'蔡琳高梓淇离婚']
    #test_labels=['川航确诊飞行员在江油密接均为阴性','狗仔镜头下的关晓彤']
    
    for label in test_labels:
        '''
        for i in range(100):
            if(a[label][d.index[i]+3]!=0):
                train_set=np.insert(train_set,0,np.array([b[label][b.index[i]],c[label][c.index[i]],d[label][d.index[i]],a[label][d.index[i]+3]]),axis=0)
        '''
        for i in dataSet.index:
            if(i==0):
                continue
            #if((i%2!=0)&(dataSet[label][i+6]!=0)&(dataSet[label][i+6]!=-1)&(dataSet[label][i+6]!=1)):
            if((i%2!=0)&(dataSet[label][i+6]!=0)):
                train_set=np.insert(train_set,0,np.array([dataSet[label][i],dataSet[label][i+2],dataSet[label][i+4],dataSet[label][i+6]])*100,axis=0)
            if(i==111):
                break
                
    train_set=pd.DataFrame(train_set)
    train_set.to_csv('Train_set.csv',encoding='utf_8_sig')
    return train_set

def run_model():
    #运行神经网络模型，并返回预测对象
    Train_set=load_Train_set()
    data=Train_set.iloc[:,:-1]
    target=np.squeeze(Train_set.iloc[:,-1:].values)
    
    X_train,X_test,Y_train,Y_test=train_test_split(data,target,test_size=0.1)
    MLPlr=MLPRegressor(activation='logistic',hidden_layer_sizes=(10,15),max_iter=200000,learning_rate='adaptive')
    MLPlr.fit(X_train,Y_train)
    
    MLP_y_predict=MLPlr.predict(X_test)
    plot_predic(MLP_y_predict,Y_test)
    #MLP_y_predict=MLPlr.predict(data)
    #plot_predic(MLP_y_predict,target)
    
    return MLPlr

def plot_predic(MLP_y_predict,Y_test):
    #画出源数据和预测数据
    plt.rcParams['font.sans-serif']='SimHei'
    
    plt.rcParams['axes.unicode_minus']=False
    fig=plt.figure(figsize=(50,50),dpi=100)
    ax=plt.subplot(1,1,1)
    ax.plot(MLP_y_predict,color='r',label='predict')
    ax.plot(Y_test,color='c',linestyle='--',label='actual')
    plt.title('神经网络模型拟合图示')
    plt.legend()
    plt.show()
    
    
def load_Train_set():
    return pd.read_csv('Train_set.csv',encoding='utf_8_sig',index_col=0)



    
    #plt.subplots_adjust(top=0.85,bottom=0.12,left=0.2,right=0.71,hspace=0,wspace=0)


def draw_predict_onLive(ax,heat,label,Model):
    #对网络爬取的数据及机器学习得出的预测值进行实时展示
    #a=pd.read_csv('wb_heat_120.csv',encoding='utf_8_sig',index_col=0)
    #df=a[ '安徽一批国产鸡腿外包装核酸阳性']
    df=pd.DataFrame()
    df[label]=heat
    df=df[label]
    data=[]
    s_predic=pd.Series([0])
    #fig, ax = plt.subplots()
    #df.plot()
    predict_stack=[]
    x=[]
    start=6
    #temp=df[start]
    ax.cla()
    for i in df.index:
        s_predic[i]=df[i]
        if(len(s_predic)>start):
            pct=s_predic.pct_change()
            pct.fillna(0,inplace=True)
            pct[np.isinf(pct)]=1
            index=pct.index[-1]
            predict_rate=Model.predict([np.array([pct[index-6],pct[index-4],pct[index-2]])])
            predict_stack.append(df[i-2]*(100+predict_rate)/100)
        else:
            predict_stack.append(float(df[i]))
            
    ax.set_title('实时预测微博热搜')
    ax.plot(heat,color='r',label='实际热度')
    ax.plot(predict_stack,color='c',linestyle='--',label='预测热度')
    ax.legend()
    ax.set_xlim(0,240)
    
  

#df,data,pct=draw_predict_onLive('',MLPlr)

def main():
    i=0
    MLPlr=run_model()
    df_heat=pd.DataFrame()
    #print('df结构:{}'.format(df_heat.shape))
    
    
    heat,labels,ticks=get_data()
    df_heat=insert(df_heat,heat,labels,ticks)
    
    plt.rcParams['font.sans-serif']='SimHei'
    plt.rcParams['axes.unicode_minus']=False
    fig=plt.figure(figsize=(50,50),dpi=100,facecolor='grey')
    gs=gridspec.GridSpec(3,2)
    
    
    ax1=plt.subplot(gs[0,0])
    ax2=plt.subplot(gs[1,0],projection='3d')
    ax3=plt.subplot(gs[0,1],projection='3d')
    ax4=plt.subplot(gs[1,1])
    ax5=plt.subplot(gs[2,:])
    Last_topic=labels[-1]
    Top_topic=''
    Top_heat=[]
    '''
    ax2=plt.subplot(gs[1,:])
    ax3=plt.subplot(gs[2,:])
    ax4=plt.subplot(gs[3,:])
    ax5=plt.subplot(gs[4,:])
    '''
    #总热度指数
    while(True):
        
        try:
            
            if(i>120):break #设定爬取120次
            
            heat,labels,ticks=get_data()
            ax1.cla()
            df_heat=insert(df_heat,heat,labels,ticks)
            y=np.array(heat)
            x1=np.arange(1,len(y)+1)
            x=np.arange(0,len(y))
            
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels,rotation=80,fontsize=8)
            #print(help(ax1.set_xticks))

            
            #ax1.xaxis.set_major_locator(MultipleLocator(2))
            #print(ax1.xaxis.set_ticklabels(labels,minor=False,rotation=80))
            #ax1.xaxis.set_ticklabels(labels,minor=False)
            #ax1.xaxis.set_label_text('123')
            #break
            #help(ax1)
            #ax1.xaxis.majorTicks=labels
            #(x,labels,rotation=90)
            ax1.bar(x,y,0.5,hatch='/')
            ax1.set_title('微博热搜条形图')
            #plt.xticks(x,labels,rotation=80)

            '''
            ax2=plt.subplot(gs[1,:])
            ax2.cla()
            ax2.bar(x,y,0.5,hatch='/')
            ax2.xaxis.set_data_interval(1,26,ignore=True)
            #ax2.xaxis.set_major_locator(MultipleLocator(1))
            ax2.xaxis.set_units([1,2,3])
            #ax2.xaxis.set_ticklabels(labels,minor=False,rotation=80)
            plt.xticks(x,labels)
            print('2222222')
            '''
            cm=plt.cm.get_cmap('RdYlBu_r')
            
            
            
            
            


            ax3.cla()
            ax3.scatter(x,x,heat,c=heat,cmap=cm,s=50)
            #ax3.set_axis_off()
            
            
            ax3.set_zticks([])
            ax3.set_title('热度散点图')
            for i in range(len(x)):
                if(i>20):
                    ax3.text(x[i],x[i],heat[i],labels[i],fontsize=8,rotation=45)
                else:
                    ax3.text(x[i],x[i],heat[i],labels[i],fontsize=5)

            Top_topic=labels[-1]
            if(Top_topic==Last_topic):
                Top_heat.append(heat[-1])
                draw_predict_onLive(ax4,Top_heat,Top_topic,MLPlr)
            else:
                Top_heat.clear()
            Last_topic=Top_topic
            Top_topic=''
            

            ax2.cla()
            x_heat,y_heat=np.array(heat),np.array(heat)
            X, Y = np.meshgrid(x_heat, y_heat)
            N=height(X,Y)
             # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)

            ax2.set_title('热搜三维图')
            ax2.plot_surface(X,Y,N, rstride=1, cstride=1, cmap=cm)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_zticks([])
            for i in range(len(x_heat)):
                if(i<22):
                
                    ax2.text(x_heat[i],y_heat[i],x_heat[i]**2+y_heat[i]**2,labels[i],fontsize=4)
                else:
                    ax2.text3D(x_heat[i],y_heat[i],x_heat[i]**2+y_heat[i]**2,labels[i],fontsize=8,rotation=45)


            ax5.cla()
            ax5.set_title('热搜热力图')
            ax5.contourf(X,Y,N,10,cmap='summer')
            for i in range(len(x)):
                if(i<22):
                
                    ax5.text(x_heat[i],y_heat[i],labels[i],fontsize=4)
                else:
                    ax5.text(x_heat[i],y_heat[i],labels[i],fontsize=8,rotation=20)

            plt.subplots_adjust(top=0.93 ,hspace=0.35,bottom=0.05)#给各个子图设定参数
            
            

            plt.pause(5)
            
            #time.sleep(60)
            i=i+1
        except Exception as e:
            print(e)
            break
            #continue
        
    df_heat.fillna(0,inplace=True)
    
    
    dataSet=make_dataSet(df_heat)
    #one_df_heat_mean.dropna(inplace=True)
    return df_heat



a=main()

'''
b=a.rolling(window=2).mean()
c=a.rolling(window=5).mean()
d=a.rolling(window=7).mean()

a.to_csv('wb_heat_120.csv',encoding='utf_8_sig')
b.to_csv('two_mean.csv',encoding='utf_8_sig')
c.to_csv('five_mean.csv',encoding='utf_8_sig')
d.to_csv('seven_mean.csv',encoding='utf_8_sig')
'''


    
