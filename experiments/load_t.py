import numpy
import mido
from mido import MidiFile
import os

path = '../files'  #change to files path
file_count = 268 #number of midi files in folder

white_list=['note_on']#type of messages accepted

inst_lib = [] #global variable containing the instrument library used in the midi set

lower_bound = 30 #discard files with less than x valid messagges
upper_bound = 60 #discard files with more than x valid messagges

#set lower_bound = upper_bound to have fixed length input (only in wrapped mode)

rolling_total = False #switch between absolute and relative time (false=relative)

#uncomment the prints for some metrics

#run pre_pros for the complete process. Other functions might be still useful if we decide on something else

def load(): #load files from folder

    i=0
    f=[]
    f=[0 for x in range(file_count)]
    directory = os.fsencode(path)
        
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mid"): 
            #print("found file " + filename)
            f[i] = MidiFile(path+'/'+filename)
            i+=1
            continue
        else:
            continue
    print('loaded ' + str(i) + ' files')
    return f

def parse(input): #transform track into list of (note,time)

    structure = []
    total=0
    for msg  in input:
        #print(msg.type)
        if(valid(msg.type)):
            if(rolling_total):
                total+=msg.time
                structure.append([msg.note,total])
            else:
                structure.append([msg.note,msg.time])
            #print(msg.type + ' ' + str(msg.note) + ' ' + str(msg.time))

    #print('n of messagges: ' + str(len(structure)))
    return structure

def valid(msg): #check if input message is valid

    for m in white_list:
        if msg == m:
            return True
    return False

def lib_gen(input): #generate instrument library from track list

    x=[]
    x=[0 for x in range(127)]

    for track in input:
        for h, tup in enumerate(track):
            #print(str(tup))
            x[tup[0]]+=1
    for i in range(127):
        if x[i]!=0:
            inst_lib.append(i)

def lib_match(a): #chack matching instrument in library

    for i,x in enumerate(inst_lib):
        if inst_lib[i]==a:
            return i
    print('library error')
    return 1
    
def wrap(input): #generate vectorized input from message list and instrument dictionary

    result = []

    for i, tup in enumerate(input):
        x=[]
        x = [0 for x in range(len(inst_lib)+1)]
        x[0] = input[i][1] #time in slot 0
        x[lib_match(input[i][0])] = 1  #instrument played at 1, rest at 0
        result.append(x)

    return result

def timeline(input): #extends a wrapped array into a timeseries. bigger list index is time

    result =[]

    x = [0 for x in range(len(inst_lib))]

    for i, a in enumerate(input):
        if a[0] != 0:
            result.append(x)
            x=[0 for x in range(len(inst_lib))]
        for h in range(len(inst_lib)):
            x[h] += a[h+1]
        for h in range(a[0]):
            result.append([0 for x in range(len(inst_lib))])

    return result




def pre_pros(): #run this

    files = load()
    v = []
    min_len = 10000
    max_len = 0

    for h, sample in enumerate(files):
        for i, track in enumerate(sample.tracks):
            #print('Track {}: {}'.format(i, track.name))
            v.append(parse(track))
            l = len(v[len(v)-1])
            if l > upper_bound or l < lower_bound:
                v.pop()
            else:
                if l < min_len:
                    min_len = l
                if l > max_len:
                    max_len = l
    #print(str(v[0]))
    print('selected tracks: ' + str(len(v)))
    #print('shortest file: ' + str(min_len))
    #print('longest file: ' + str(max_len))
    lib_gen(v)#necessary to call wrap() and timeline()
    print('instrument dictionary' +  str(inst_lib))
    wv=[] #wrapped input
    twv=[] #timeline input
    for i, a in enumerate(v):
        wv.append(wrap(a))
    print(str(wv[0]))
    for i, a in enumerate(wv):
        twv.append(timeline(a))
    print(str(twv[0]))

