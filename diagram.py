import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,4))
ax.axis('off')

# Input nodes
input_nodes = [(0,4),(0,3)]
for x,y in input_nodes:
    ax.scatter(x,y,s=500,color='skyblue')
ax.scatter(0,2,s=500,color='lightgreen') # bias

# Hidden nodes
hidden_nodes = [(2,5),(2,4.2),(2,3.4),(2,2.6),(2,1.8)]
for x,y in hidden_nodes:
    ax.scatter(x,y,s=500,color='orange')

# Output node
ax.scatter(4,3,s=500,color='red')

# Connections
for xi,yi in input_nodes + [(0,2)]:
    for hx,hy in hidden_nodes:
        ax.plot([xi,hx],[yi,hy],'k--',alpha=0.5)

for hx,hy in hidden_nodes:
    ax.plot([hx,4],[hy,3],'k--',alpha=0.5)

# Labels
ax.text(-0.2,4,'x1',fontsize=12)
ax.text(-0.2,3,'x2',fontsize=12)
ax.text(-0.2,2,'1(bias)',fontsize=12)
for i,(hx,hy) in enumerate(hidden_nodes):
    ax.text(hx-0.2,hy+0.1,f'h{i+1}',fontsize=12)
ax.text(4.2,3,'o',fontsize=12)

plt.show()
