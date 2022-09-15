from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

## %%%%%%%%%%%%%%% Your code here - Begin %%%%%%%%%%%%%%%
classifier = LinearDiscriminantAnalysis()
classifier.fit(x_train_shaped,y_train_shaped.ravel())
h_func = lambda x: classifier.predict(x.reshape(x.shape[0], -1))
## %%%%%%%%%%%%%%% Your code here - End %%%%%%%%%%%%%%%%%

print(f'The score on the validation set is: {calc_score(h_func, x_val, y_val):.4f}')