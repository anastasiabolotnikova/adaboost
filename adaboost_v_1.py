import numpy as np
from math import log
import matplotlib.pyplot as plt
import cv2
import os
import time

## Adaboost is use
window_size = 10

class Haar(object):
    def __init__(self, type, feature, size, shape, start):
        self.type=type
        self.feature = cv2.resize(feature, (shape[1]*size, shape[0]*size), interpolation=cv2.INTER_NEAREST)
        self.start = start
        self.size=size
        self.shape = shape

class WeakClassifier(object):
    def __init__(self, haar, theta, sign, weight):
        self.haar = haar
        self.theta = theta
        self.sign = sign
        self.weight = weight

# 12*12 subwindow
# haar has (i,j - start) and size
def get_haar_score(haar, subwindow):
    #print(haar.feature)
    #print(subwindow[haar.start[0]:haar.start[0]+haar.size*haar.shape[0],
                               #haar.start[1]:haar.start[1]+haar.size*haar.shape[1]])
    score=sum(sum(haar.feature*subwindow[haar.start[0]:haar.start[0]+haar.size*haar.shape[0],
                               haar.start[1]:haar.start[1]+haar.size*haar.shape[1]]))
    return score


def feature_weighted_error_rate(actual, predicted, weights):
    return sum(weights*(np.not_equal(actual, predicted)))


def predict(score, classifier):
    if score<classifier.theta:
        return -classifier.sign
    return classifier.sign


def display(fovea):
    plt.imshow(fovea, interpolation='nearest')
    plt.show()


def get_positive_samples(path):
    global window_size
    positive_samples = []
    for file in os.listdir(path):
        if not file.endswith(".jpg"):
            positive_samples.append(cv2.resize(np.loadtxt(path + file), (window_size,window_size), interpolation=cv2.INTER_NEAREST))
    return positive_samples

def get_nagetive_samples(labeled_set_file, path2samples):
    global window_size
    negative_samples = []
    f = open(labeled_set_file)
    files = f.readlines()
    for file in files:
        line = file.split("^J")
        coords =  (line[-1]).split(")(")

        fovea_grey = []
        f=open(path2samples + line[0]+"^J")
        line=f.readline()
        while line != "grayscale\n":
            line=f.readline()
        line=f.readline()
        while line!="Edges\n":
            fovea_grey.append([int(el) for el in line.split()])
            line=f.readline()
        #display(fovea_grey)
        fovea_grey = np.array(fovea_grey)

        for i in range(len(coords)):
            coords[i] = coords[i].strip(" ()\n")

        if len(coords)>1:
            a,b,c,d = [int(coord) for coord in (coords[-2] + "," + coords[-1]).split(",")]
            if b<d and a<c:

                #display(fovea_grey[b:d, a:c])
                #negative_samples.append(cv2.resize(fovea_grey[b:d, a:c], (10,10), interpolation=cv2.INTER_NEAREST))
                for i in [x for x in range(0, fovea_grey.shape[0]-window_size, window_size/2) if x not in range(b-window_size/2, d+window_size/2)]:
                    for j in [x for x in range(0, fovea_grey.shape[1]-window_size, window_size/2) if x not in range(a-window_size/2, c+window_size/2)]:
                        negative_sample = fovea_grey[i:i+window_size, j:j+window_size]
                        #display(negative_sample)
                        negative_samples.append(negative_sample)

    return negative_samples

# Generate positive samples
positive_samples = get_positive_samples("code_cc/training+/")
#for i in range(len(positive_samples)):
#    plt.imshow(positive_samples[i], interpolation='nearest', cmap='Greys_r')
#    plt.savefig("positives/"+str(i))
positive_samples_rotated = [np.rot90(np.rot90(np.rot90(el))) for el in positive_samples]
positive_samples = positive_samples + positive_samples_rotated

# Generate Negative samples
negative_samples = get_nagetive_samples("code_cc/output", "code_cc/cc/")
#negative_samples_rotated = [np.rot90(np.rot90(np.rot90(el))) for el in negative_samples]
#negative_samples = negative_samples + negative_samples_rotated

# Shuffle all the samples
np.random.shuffle(positive_samples)
np.random.shuffle(negative_samples)

print("Tatal num of samples at our disposal:")
print("Positives: " + str(len(positive_samples)))
print("Negatives: " + str(len(negative_samples)))

split = 0.95
pos_split = int(len(positive_samples)*split)
neg_split = int(len(negative_samples)*split)

training_set = positive_samples[0:pos_split] + negative_samples[0:neg_split]
testing_set = positive_samples[pos_split:] + negative_samples[neg_split:]

nrPos = pos_split
nrNeg = neg_split

nrPos_test = len(positive_samples)-nrPos
nrNeg_test = len(negative_samples)-nrNeg

training_labels = [1]*nrPos + [-1]*nrNeg
testing_labels = [1]*nrPos_test + [-1]*nrNeg_test

print("For training")
print("Positives: "+str(nrPos))
print("Negatives: "+str(nrNeg))

print("For testing")
print("Positives: "+str(nrPos_test))
print("Negatives: "+str(nrNeg_test))


# Generate many haar features
features_start=[]

# Define haar feature types
haar1 = np.array([1, -1,
                  1, -1])
haar1.shape = (2,2)

haar2 = np.array([1, 1,
                  -1, -1])
haar2.shape = (2,2)

haar3 = np.array([1, -1, 1,
                  1, -1, 1])
haar3.shape = (2,3)

haar4 = np.array([1, 1,
                  -1, -1,
                  1, 1])
haar4.shape = (3,2)

haar5 = np.array([1, -1,
                  -1, 1])
haar5.shape = (2,2)

haar6 = np.array([-1, 1,
                  1, -1])
haar6.shape = (2,2)

haar7 = np.array([1, 1, 1,
                  1, -1, 1,
                  1, 1, 1])
haar7.shape = (3,3)


# Define many sizes for all feature types
haar_feature_types=[haar1,haar2,haar3,haar4,haar5,haar6, haar7]
for f in range(len(haar_feature_types)):
    shape = haar_feature_types[f].shape
    if 3 in haar_feature_types[f].shape:
        max_size=4
    else:
        max_size=7

    for s in range(1, max_size+1):
        features_start.append(Haar(f+1, haar_feature_types[f], s, shape, (0,0)))

features = []
for j in features_start:
        # Get all posible starting locations for this feature
        starting_positions = []
        space = (window_size-j.shape[0]*j.size, window_size-j.shape[1]*j.size)
        for k in range(space[0]+1):
            for l in range(space[1]+1):
                starting_positions.append((k, l))

        for loc in starting_positions:
            features.append(Haar(j.type, j.feature, j.size, j.shape, loc))


#for f in features:
#    print(f.feature)
#    print(f.start)

features = list(features)
n1 = len(set(features))
n2 = len(features)
print(n1)

feature_weights=[]
weak_classifires = []


np.random.shuffle(features)


### Cascade Creation ###

F_target = 0.001
f = 0.5

F_i = 1
#i = 0


cascade = []
start_time = time.time()

image_weights = [1.0/(2*nrNeg)]*nrNeg + [1.0/(2*nrPos)]*nrPos

show_stuff = False

while F_i > F_target:
    #i += 1

    ## Train classifier for stage i

    best_feature_index = 0
    best_weak_classifier = 0
    lowest_error = float("inf")

    #image_weights = [1.0/(2*nrNeg)]*nrNeg + [1.0/(2*nrPos)]*nrPos
    total = sum(image_weights)
    image_weights = [w / total for w in image_weights]

    f_i = 1
    cycle = 0

    while f_i > f:
        print("f_i in loop: "+str(f_i))
        total = sum(image_weights)
        if total != 1:
            image_weights = [w / total for w in image_weights]

        print()
        errors = []
        # For every feature, find best threshold and compute corresponding weighted error
        for j in features:
            posProb = 0
            avgPosScore = 0.0
            negProb = 0
            avgNegScore = 0.0
            # Apply feature to each image and get threshold for current feature (current location)
            for i in range(len(training_set)):
                score=get_haar_score(j, training_set[i])
                if training_labels[i]==1:
                    posProb +=image_weights[i]
                    avgPosScore += score
                    #avgPosScore += score*image_weights[i]
                else:
                    negProb +=image_weights[i]
                    avgNegScore += score
                    #avgNegScore += score*image_weights[i]

            avgPosScore = avgPosScore / nrPos
            #avgPosScore = avgPosScore / posProb
            avgNegScore = avgNegScore / nrNeg
            #avgNegScore = avgNegScore / negProb
            polarity = 0
            if avgPosScore>avgNegScore:
                polarity = 1
            else:
                polarity = -1

            # Optimal theta found
            theta = (avgPosScore + avgNegScore) / 2
            #print("Theta: " + str(theta))

            # Create classifier object
            w_classif = WeakClassifier(j, theta, polarity, 0)

            # Compute weighted error
            predicted = []
            for sample in training_set:
                # Get predictions of all samples
                score=get_haar_score(j, sample)
                predicted.append(predict(score, w_classif))

            weighted_error=feature_weighted_error_rate(training_labels, predicted, image_weights)
            errors.append(weighted_error)

            # Look for the lowest error and keep track of the corresponding classifier
            if weighted_error<lowest_error:
                lowest_error = weighted_error
                best_weak_classifier = w_classif
                best_feature_index = features.index(j)

        print("Best feature index: "+str(best_feature_index))

        if show_stuff:
            plt.plot(errors)
            plt.show()

        ## Choose weak classifier with lowest error ##
        beta_t = lowest_error/(1-lowest_error)

        if beta_t==0:
            inverted_weighth = 0
        else:
            inverted_weighth = log(1/beta_t)
        best_weak_classifier.weight = inverted_weighth

        ## Update weights and evaluate current weak classifier ##
        predicted=[]
        scores_debug = []
        for sample in training_set:
            # Get weighted classification error
            score=get_haar_score(best_weak_classifier.haar, sample)
            scores_debug.append(score)
            predicted.append(predict(score, best_weak_classifier))

        FP = 0.0
        FN = 0.0
        TP = 0.0
        TN = 0.0
        colors_predicted = []
        for k in range(len(image_weights)):
            # if sample is correctly classified

            if training_labels[k] == 1 and predicted[k] == -1:
                FN += 1
            if training_labels[k] == -1 and predicted[k] == 1:
                FP += 1

            # Update image weights
            if training_labels[k]==predicted[k]:
                image_weights[k] = image_weights[k]*beta_t
                if predicted[k] == 1:
                    TP += 1
                if predicted[k] == -1:
                    TN += 1

            if predicted[k]==-1:
                colors_predicted.append('r')
            else:
                colors_predicted.append('g')

        ## Evaluate f_i
        f_i = (FP/(2*nrNeg))+(FN/(2*nrPos))
        print("f_i: " + str(f_i))

        print("TP, TN, FP, FN for the current weak classifier:")
        print(TP/nrPos, TN/nrNeg, FP/nrNeg, FN/nrPos)

        ## Visualize the performace of weak classifier for training samples
        if show_stuff:
            plt.scatter(range(nrPos+nrNeg), scores_debug, c = colors_predicted)
            plt.vlines(nrPos,min(scores_debug),max(scores_debug))
            plt.plot(range(nrPos+nrNeg), [best_weak_classifier.theta]*(nrPos+nrNeg))
            plt.xlim(0,nrPos+nrNeg)
            plt.show()

        print("Threshold of the best feature: "+str(best_weak_classifier.theta))

        cycle += 1

    cascade.append(best_weak_classifier)

    print len(features)

    print(best_weak_classifier.haar.feature)

    strong_FP = 0.0
    strong_FN = 0.0

    cascade_scores = []
    cascade_colors_predicted = []
    for l in range(len(training_set)):
        strong_score = 0.0
        for w_class in cascade:
            strong_score += w_class.weight * predict(get_haar_score(w_class.haar, training_set[l]), w_class)
        cascade_scores.append(strong_score)
        clas = np.sign(strong_score)
        if clas==-1:
            cascade_colors_predicted.append('r')
        else:
            cascade_colors_predicted.append('g')

        if training_labels[l] == 1 and clas == -1:
            strong_FN += 1
        if training_labels[l] == -1 and clas == 1:
            strong_FP += 1

    ## Visualize the performace of the cascade on training samples
    if show_stuff:
        plt.scatter(range(nrPos+nrNeg), cascade_scores, c = cascade_colors_predicted)
        plt.vlines(nrPos,min(cascade_scores),max(cascade_scores))
        plt.plot(range(nrPos+nrNeg), [0]*(nrPos+nrNeg))
        plt.xlim(0,nrPos+nrNeg)
        plt.show()

    F_i = (strong_FP/(2*nrNeg))+(strong_FN/(2*nrPos))
    print("F_i: " + str(F_i))
    print("Cascade size: "+str(len(cascade)))

print("--- %s seconds ---" % (time.time() - start_time))


print("Now running cascade on the testing set")

FP_test = 0.0
FN_test = 0.0
TP_test = 0.0
TN_test = 0.0

scores = []

print("Cascade:")
print(cascade)

f_cnt=1
for el in cascade:
    print(" ")
    '''print("Start, feature, size, shape")
    print(el.haar.start)
    print(el.haar.feature)
    print(el.haar.size)
    print(el.haar.shape)
    print("Theta, weight, sign")
    print(el.theta)
    print(el.weight)
    print(el.sign)'''

    print("// Compute "+str(f_cnt)+" feature score")
    print("vote = f_vote(greyIntegral, "+str(el.haar.type)+", w, h, "+
          str(el.haar.feature.shape[1])+", "+str(el.haar.feature.shape[0])+", "+str(el.haar.start[1])+", "+str(el.haar.start[0])+", "
          +str(int(round(el.theta)))+", "+str(el.weight*el.sign)+", fovea);")
    print("cascade_score += vote;")

    f_cnt+=1

save = False

for t in range(len(testing_set)):
    strong_score = 0.0
    for w_class in cascade:
        #print("Loc: " +str(w_class.haar.start))
        strong_score += w_class.weight * predict(get_haar_score(w_class.haar, testing_set[t]), w_class)
    clas = np.sign(strong_score)
    scores.append(strong_score)

    if testing_labels[t] == 1 and clas == -1:
        FN_test += 1
        if save:
            plt.imshow(testing_set[t], interpolation='nearest')
            plt.savefig("FN/"+str(t)+".jpg")
    if testing_labels[t] == -1 and clas == 1:
        FP_test += 1
        if save:
            plt.imshow(testing_set[t], interpolation='nearest')
            plt.savefig("FP/"+str(t)+".jpg")
    if testing_labels[t] == 1 and clas == 1:
        TP_test += 1
        if save:
            plt.imshow(testing_set[t], interpolation='nearest')
            plt.savefig("TP/"+str(t)+".jpg")
    if testing_labels[t] == -1 and clas == -1:
        TN_test += 1
        if save:
            plt.imshow(testing_set[t], interpolation='nearest')
            plt.savefig("TN/"+str(t)+".jpg")

    #print(testing_labels[t]==clas)
    #print(testing_labels[t])
    #print(clas)


print(FP_test/(2*nrNeg_test))+(FN_test/(2*nrPos_test))
print(FP_test)
print(FN_test)

plt.plot(range(nrPos_test), scores[0:nrPos_test], 'go')
plt.plot(range(nrPos_test,nrPos_test+nrNeg_test), scores[nrPos_test:], 'ro')
plt.plot(range(nrNeg_test), [0]*nrNeg_test)
plt.show()

print("TP, TN, FP, FN for the cascade classifier:")
print(TP_test/nrPos_test, TN_test/nrNeg_test, FP_test/nrNeg_test, FN_test/nrPos_test)

