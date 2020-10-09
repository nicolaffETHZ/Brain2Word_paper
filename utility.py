import numpy as np
import os

def mean_Concat(Concat_pred, Concat_mean, first):
    out = np.zeros((180, Concat_pred.shape[1]))
    for i in range(180):
        values = Concat_pred[i::180,:]
        out[i,:] = np.mean(values, axis=0)
    if first:
        return out
    else:
        return (out + Concat_mean)/2

def sizes(pre_reduced):
    data_path = str(os.path.dirname(os.path.abspath(__file__))) + '/data/subjects/'
    fil = 'data_180concepts_sentences.mat'
    if pre_reduced:
        subjects = ['P01','M02','M03','M04','M05','M06','M07', 'M08', 'M09','M15'] 
    else:
        subjects = ['P01','M02','M03','M04','M05','M06','M07', 'M08', 'M09','M15','M10', 'M13','M14','M16', 'M17'] 
    sizes = np.zeros((333))
    lister = [None]*333
    for sub in subjects:
        Gordon_areas, _ = ROI_loader(gpu=False, subject=sub, fil=fil)
        for i in range(333):
            new_size = Gordon_areas[i][0].shape[0]
            # print(new_size)
            if new_size>sizes[i]:
                sizes[i] = new_size
                lister[i] = sub
    if not pre_reduced:
        np.save(file=str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/sizes', arr=sizes.astype(int))
    return sizes.astype(int)


def roundup2(x):
    return x if x % 1 == 0 else x + 1 - x % 1

def reduced_sizes(sizes):

    new = np.zeros_like(sizes)
    for i in range(sizes.shape[0]):
        new[i] = roundup2(sizes[i]/20) # was 20
    new = new.astype(int)
    for i in range(sizes.shape[0]):
        if new[i] == 0:
            new[i] = 1
    np.save(file=str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/reduced_sizes', arr=new.astype(int))

def coltocoord_ROI_ordering(subject, fil):

    data_path = str(os.path.dirname(os.path.abspath(__file__))) +  '/data/subjects/'

    
    all_data = sio.loadmat(data_path + subject + '/' + fil)        
    ROI = all_data['meta']
    coord = ROI[0][0][5]
    return coord

def ROI_loader(subject, fil):

    data_path = str(os.path.dirname(os.path.abspath(__file__))) + '/data/subjects/'
    
    all_data = sio.loadmat(data_path + subject + '/' + fil)       
    ROI = all_data['meta']
    Gordon_areas = ROI[0][0][11][0][14]  
    try:
        data = all_data['examples']
    except KeyError:
        data = all_data['examples_passagesentences']
    return Gordon_areas, data


def last_dim():
    subjects = ['P01','M02','M03','M04','M05','M06','M07', 'M08', 'M09','M15','M10', 'M13','M14','M16','M17']

    container = []

    for q in range(333):
        ROI_region = q
        counter = 0
        for i, sub in enumerate(subjects):
            
            ROIs, data = ROI_loader(False,sub,'data_180concepts_pictures.mat')
            coord = coltocoord_ROI_ordering(False,sub,'data_180concepts_pictures.mat')
            grid = np.zeros((88,125,85))
            Roi_coord = np.squeeze(coord[ROIs[ROI_region][0]])
            assert Roi_coord.shape[0] == ROIs[ROI_region][0].shape[0]
            if counter < np.squeeze(Roi_coord).shape[0]:
                last_dim = np.asarray(np.unique(np.squeeze(Roi_coord)[:,2], return_counts=True))
                assert np.sum(last_dim[1,:]) == ROIs[ROI_region][0].shape[0]
                counter = np.squeeze(Roi_coord).shape[0]
                subber = sub
        container.append(last_dim)
    np.savez(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/last_dim', *container)



def look_ups():
    if  not os.path.exists(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/sizes.npy'):
        sizes(False)
    if not os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/reduced_sizes.npy'):
        reduced_sizes(sizes(True))
    if not os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/last_dim.npz'):
        last_dim
    print('Look up tables were created')

def files_exist():
    booler = os.path.exists(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/sizes.npy')
    booler = booler and os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/reduced_sizes.npy')
    booler = booler and os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/last_dim.npz')
    return booler





##################################################### Evaluation functions #######################################################


def top_5(pred, real):
    counter = 0.0
    counterr = 0.0
    counter_ten= 0.0
    for i in range(pred.shape[0]):
        if np.argmax(pred[i,:])==np.argmax(real[i,:]):
            counter+=1
        sort = np.flip(np.argsort(pred[i,:]))
        holder = np.isin(np.argmax(real[i,:]),sort[:5])
        holder_ten = np.isin(np.argmax(real[i,:]),sort[:10])
        if holder:
            counterr+=1
        if holder_ten:
            counter_ten+=1
    accuracy = counter/pred.shape[0]
    accuracy_five = counterr/pred.shape[0]
    accuracy_ten = counter_ten/pred.shape[0]
    return accuracy, accuracy_five, accuracy_ten

def evaluation(vectors_real,vectors_new):

    count = 0
    total = 0
    for i in range(vectors_real.shape[0]):
        for j in range(vectors_real.shape[0]):
            if j>i:
                errivsi = np.corrcoef(vectors_new[i,:],vectors_real[i,:])
                errivsj = np.corrcoef(vectors_new[i,:],vectors_real[j,:])
                errjvsi = np.corrcoef(vectors_new[j,:],vectors_real[i,:])
                errjvsj = np.corrcoef(vectors_new[j,:],vectors_real[j,:])

                if (errivsi[0,1] + errjvsj[0,1]) > (errivsj[0,1] + errjvsi[0,1]):
                    count+=1
                total+=1

    accuracy = count/total
    return accuracy