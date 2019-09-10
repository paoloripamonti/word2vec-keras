from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from word2vec_keras import Word2VecKeras

# fetch the dataset using scikit-learn
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

train_b = fetch_20newsgroups(subset='train',
                             categories=categories, shuffle=True, random_state=42)
test_b = fetch_20newsgroups(subset='test',
                            categories=categories, shuffle=True, random_state=42)

print('size of training set: %s' % (len(train_b['data'])))
print('size of validation set: %s' % (len(test_b['data'])))
print('classes: %s' % (train_b.target_names))

x_train = train_b.data
y_train = [train_b.target_names[idx] for idx in train_b.target]
x_test = test_b.data
y_test = [train_b.target_names[idx] for idx in test_b.target]

model = Word2VecKeras()
model.train(x_train, y_train, w2v_epochs=20, k_epochs=1)

pprint(model.evaluate(x_test, y_test))

model.save()
