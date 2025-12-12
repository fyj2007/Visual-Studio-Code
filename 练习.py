def a(q, l, r):
    if l==r:
        return
    i, j = l+1, r+1
    x = q[(l+r)//2]
    while  i < j:
        while True:
            i+=1
            if q[i] >= x:
                break
        while True:
            j-=1
            if q[j] <= x:
                break
        if i < j:
            q[i],q[j]=q[j],q[i]
    a(q,l,j)
    a(q,j+1,r)
n = int(input())
q = list(map(int,input().split()))
a(q,0, n-1)
for i in q:
    print(i,end=' ')

   