#!/usr/bin/env python
# coding: UTF-8

##################################################################
#2019_2_9
#テオヤンセンストランドビースト　リンク計算用
##################################################################

import rospy
import math
import numpy as np
import sys
import time
import matplotlib.pyplot as plt


### define ###
a = 38.0
b = 41.5
c = 39.3
d = 40.1
e = 55.8
f = 39.4
g = 36.7    
h = 65.7
i = 49.0
j = 50.0
k = 61.9
l = 7.8
m = 15.0
O = np.array([0.0,0.0])  


def rtod(rad_):
    deg = rad_*180.0/math.pi
    return deg


def dtor(deg_):
    rad = deg_ * math.pi /180
    return rad

    
def calc_link(theta_deg_):
    t_A = dtor(theta_deg_)

    plt.axes().set_aspect('equal','datalim') #アスペクト比を1:1に変更
    plt.xlim([-150,50])
    plt.ylim([-100,50])
    
    ### calc A point ### 
    A = np.array([0.0,0.0])
    A[0] = m * math.cos(t_A)
    A[1] = m * math.sin(t_A)
    
    ### calc B point ###
    B = np.array([0.0,0.0])
    B[0] = (O[0] - a)
    B[1] = (O[1] - l)
    
    ### calc C point ###
    C = np.array([0.0,0.0])
    AB   = math.sqrt((a+A[0])**2+(l+A[1])**2)
    t_B  = math.acos( (b**2 + AB**2 -(j**2)) / (2*b*AB))
    t_AB = math.atan( (A[1]-B[1]) / (A[0]-B[0]) )
    C[0] = B[0] + b*math.cos( t_B + t_AB )
    C[1] = B[1] + b*math.sin( t_B + t_AB )
    
    ### calc D point ###
    D = np.array([0.0,0.0])
    t_BD = math.acos( (AB**2 + c**2 -(k**2)) / (2*AB*c) ) - t_AB
    D[0] = B[0] + c*math.cos(t_BD)
    D[1] = B[1] - c*math.sin(t_BD)
    
    ### calc E point ###
    E = np.array([0.0,0.0])
    t_CBE = math.acos( (b**2 + d**2 -(e**2)) / (2*d*b))
    t_BE  = math.pi - (t_CBE + t_B + t_AB)
    E[0] = B[0] - d*math.cos(t_BE)
    E[1] = B[1] + d*math.sin(t_BE)            
    
    ### calc F point ###
    F = np.array([0.0,0.0])
    t_EBD = 2*math.pi - (t_CBE + t_B + t_AB + t_BD)
    ED = math.sqrt( d**2 + c**2 -2*d*c*math.cos(t_EBD))
    t_EDF = math.acos( (ED**2 + g**2 - (f**2)) / (2*ED*g))
    t_ED = math.atan((E[1] - D[1]) / math.fabs(E[0] - D[0]))
    t_g = t_ED - t_EDF
    F[0] = D[0] - g*math.cos(t_g)
    F[1] = D[1] + g*math.sin(t_g)
    
    ### calc G point ###
    G = np.array([0.0,0.0])
    t_FDG = math.acos( (g**2 + i**2 -(h**2)) / (2*g*i))
    t_FD = t_FDG - t_g
    G[0] = D[0] - i*math.cos(t_FD)
    G[1] = D[1] - i*math.sin(t_FD)
    
    # plt.scatter(O[0],O[1],color=(1,0,0))   #赤
    # plt.scatter(A[0],A[1],color=(0,1,0))   #緑
    # plt.scatter(B[0],B[1],color=(1,0,0))   #赤
    # plt.scatter(C[0],C[1],color=(1,0.5,0)) #オレンジ
    # plt.scatter(D[0],D[1],color=(0,1,1))   #水色
    # plt.scatter(E[0],E[1],color=(0.5,0,1)) #紫
    # plt.scatter(F[0],F[1],color=(0,0,0))   #黒
    plt.scatter(G[0],G[1],color=(0,0,1))   #青

    plt.plot([O[0],A[0]],[O[1],A[1]],color=(0,0,1))
    plt.plot([A[0],C[0]],[A[1],C[1]],color=(1,0,0))
    plt.plot([A[0],D[0]],[A[1],D[1]],color=(1,0,0))
    plt.plot([C[0],B[0]],[C[1],B[1]],color=(1,0,0))
    plt.plot([C[0],E[0]],[C[1],E[1]],color=(1,0,0))
    plt.plot([B[0],D[0]],[B[1],D[1]],color=(1,0,0))
    plt.plot([E[0],F[0]],[E[1],F[1]],color=(1,0,0))
    plt.plot([E[0],B[0]],[E[1],B[1]],color=(1,0,0))
    plt.plot([F[0],D[0]],[F[1],D[1]],color=(1,0,0))
    plt.plot([F[0],G[0]],[F[1],G[1]],color=(1,0,0))
    plt.plot([D[0],G[0]],[D[1],G[1]],color=(1,0,0))
    
    a_ = a
    b_ = np.linalg.norm(C-B)
    c_ = np.linalg.norm(B-D)
    d_ = np.linalg.norm(E-B)
    e_ = np.linalg.norm(C-E)
    f_ = np.linalg.norm(E-F)
    g_ = np.linalg.norm(F-D)
    h_ = np.linalg.norm(F-G)
    i_ = np.linalg.norm(D-G)
    j_ = np.linalg.norm(C-A)
    k_ = np.linalg.norm(A-D)
    l_ = l
    m_ = np.linalg.norm(A-O)

    print "O",O
    print "A",A
    print "B",B
    print "C",C
    print "D",D
    print "E",E
    print "F",F
    print "G",G

    # print "a",a,a_
    # print "b",b,b_
    # print "c",c,c_
    # print "d",d,d_
    # print "e",e,e_
    # print "f",f,f_
    # print "g",g,g_
    # print "h",h,h_
    # print "i",i,i_
    # print "j",j,j_
    # print "k",k,k_
    # print "l",l,l_
    # print "m",m,m_
    
    plt.pause(0.01)

    
if __name__ == '__main__':

    for kkk in range(10):
        for theta_deg_ in range(18):
            theta_deg = theta_deg_ * 20.0
            plt.clf()
            calc_link(float(theta_deg))

    plt.show()
    print "end"
