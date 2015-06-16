import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 4))
ax = fig.add_axes([0, 0, 0.8, 1], frameon=False, xticks=[], yticks=[])
ax.set_title('Example Decision Tree: Classification', size=24)

def text(ax, x, y, t, size=20, **kwargs):
    ax.text(x, y, t,
                ha='center', va='center', size=size,
                bbox=dict(boxstyle='round', ec='k', fc='w'), **kwargs)

text(ax, 0.5, 0.9, "How big are you?", 20)
text(ax, 0.3, 0.6, "Troll", 18)
text(ax, 0.7, 0.6, "Are you blue?", 18)
text(ax, 0.62, 0.3, "Smurf", 14)
text(ax, 0.88, 0.3, "Fairy", 14)

text(ax, 0.4, 0.75, "> 1m", 12, alpha=0.4)
text(ax, 0.6, 0.75, "< 1m", 12, alpha=0.4)
text(ax, 0.66, 0.45, "yes", 12, alpha=0.4)
text(ax, 0.79, 0.45, "no", 12, alpha=0.4)

ax.plot([0.3, 0.5, 0.7], [0.6, 0.9, 0.6], '-k')
ax.plot([0.62, 0.7, 0.88], [0.3, 0.6, 0.3], '-k')
ax.axis([0, 1, 0, 1])

plt.savefig('illustration.png')
plt.close()


def bouncer(size, color):
    if (size >= 1):
       return "Troll"
    else:
        if (color == "blue"):
           return "Smurf"
        else:
           return "Fairy"

print bouncer(1, "green")
print bouncer(0.5, "blue")
print bouncer(0.2, "pink")


from random import randint, gauss
from numpy import asarray

data = [[gauss(0.5, 0.2), 1] for i in range(10)] + [[gauss(0.3, 0.2), 0] for i in range(10)]+[[gauss(0.8, 0.2), 0] for i in range(10)] 
labels = asarray(['Smurf']*10 + ['Fairy']*10 + ['Troll']*10) 

from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, labels)

from sklearn.externals.six import StringIO  
import pydot 
dot_data = StringIO() 
tree.export_graphviz(clf, feature_names=["Size", "Color"], out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("test.pdf")

print clf.predict([[1,0],[0.5,1],[0.2,0]])

data = [[gauss(0.5, 0.2), 1] for i in range(1000)] + [[gauss(0.3, 0.2), 0] for i in range(1000)]+[[gauss(0.8, 0.2), 0] for i in range(1000)]
labels = asarray(['Smurf']*1000 + ['Fairy']*1000 + ['Troll']*1000)

Xtrain, Xtest, ytrain, ytest = train_test_split(data, labels, random_state=0)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xtrain, ytrain)

scores = cross_val_score(clf, Xtest, ytest)
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)


data = [[gauss(0.5, 0.2), 1] for i in range(1000)] + [[gauss(0.3, 0.2), 0] for i in range(1000)]+[[gauss(0.8, 0.2), 0] for i in range(1000)]
labels = asarray(['Smurf']*1000 + ['Fairy']*1000 + ['Troll']*1000)

from sklearn.ensemble import RandomForestClassifier


Xtrain, Xtest, ytrain, ytest = train_test_split(data, labels, random_state=0)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(Xtrain, ytrain)

print rfc.predict([[1,0],[0.5,1],[0.2,0]])
scores = cross_val_score(rfc, Xtest, ytest, cv=100)
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

