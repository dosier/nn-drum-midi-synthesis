import numpy
import mido
from mido import MidiFile
import os
import music21
from music21 import converter
from music21.stream import Stream




class input_block:

	def __init__(self,default, file_path, low_bound, up_bound, quantize_resolution):# if default==true, use default valuse, else use given


		if default:

			self.path = './files' 
			self.lower_bound = 0 #discard files with less than x valid messagges
			self.upper_bound = 9999999 #discard files with more than x valid messagges
			self.resolution = 16
		else:
			self.path = file_path
			self.lower_bound = low_bound
			self.upper_bound = up_bound
			self.resolution = quantize_resolution

		self.inst_lib = [] #variable containing the instrument library used in the midi set
		self.white_list=['note_on']#type of messages accepted
		self.rolling_total = False #switch between absolute and relative time (false=relative)
		self.v = []
		self.wv = []
		self.twv = []

		return

	# private functions #

	def load(self): #load files from folder

		i=0
		f=[]
		directory = os.fsencode(self.path)
			
		for file in os.listdir(directory):
			filename = os.fsdecode(file)
			if filename.endswith(".mid"): 
				#print("found file " + filename)
				f.append(MidiFile(self.path+'/'+filename))
				i+=1
				continue
			else:
				continue
		print('loaded ' + str(i) + ' files')
		return f

	def quantize_folder(self):

		directory = os.fsencode(self.path)
			
		for file in os.listdir(directory):
			filename = os.fsdecode(file)
			if filename.endswith(".mid"): 
				#print("found file " + filename)
				m = converter.parse(self.path+'/'+filename)
				m = m.quantize([self.resolution])
				m.write('midi', self.path +'/q'+filename)
				continue
			else:
				continue

		return



	def parse(self,input): #transform track into list of (note,time)

		structure = []
		total=0
		for msg  in input:
			#print(msg.type)
			if(self.valid(msg.type)):
				if(self.rolling_total):
					total+=msg.time
					structure.append([msg.note,total])
				else:
					structure.append([msg.note,msg.time])
				#print(msg.type + ' ' + str(msg.note) + ' ' + str(msg.time))

		#print('n of messagges: ' + str(len(structure)))
		return structure

	def valid(self,msg): #check if input message is valid

		for m in self.white_list:
			if msg == m:
				return True
		return False

	def lib_gen(self,input): #generate instrument library from track list

		x=[]
		x=[0 for x in range(127)]

		for track in input:
			for h, tup in enumerate(track):
				#print(str(tup))
				x[tup[0]]+=1
		for i in range(127):
			if x[i]!=0:
				self.inst_lib.append(i)
		return

	def lib_match(self,a): #chack matching instrument in library

		for i,x in enumerate(self.inst_lib):
			if self.inst_lib[i]==a:
				return i
		print('library error')
		return 1
		
	def wrap(self,input): #generate vectorized input from message list and instrument dictionary

		result = []

		for i, tup in enumerate(input):
			x=[]
			x = [0 for x in range(len(self.inst_lib)+1)]
			x[0] = input[i][1] #time in slot 0
			x[self.lib_match(input[i][0])] = 1  #instrument played at 1, rest at 0
			result.append(x)

		return result

	def timeline(self,input): #extends a wrapped array into a timeseries. bigger list index is time

		result =[]

		x = [0 for x in range(len(self.inst_lib))]

		for i, a in enumerate(input):
			if a[0] != 0:
				result.append(x)
				x=[0 for x in range(len(self.inst_lib))]
			for h in range(len(self.inst_lib)):
				x[h] += a[h+1]
			for h in range(a[0]):
				result.append([0 for x in range(len(self.inst_lib))])

		return result

	def generate_v(self):

		files = self.load()

		min_len = 10000
		max_len = 0

		for h, sample in enumerate(files):
			for i, track in enumerate(sample.tracks):
				print('Track {}: {}'.format(i, track.name))
				self.v.append(self.parse(track))
				l = len(self.v[len(self.v)-1])
				if l > self.upper_bound or l < self.lower_bound:
					self.v.pop()
				else:
					if l < min_len:
						min_len = l
					if l > max_len:
						max_len = l
		#print(str(self.v[0]))
		print('selected tracks: ' + str(len(self.v)))
		print('shortest file: ' + str(min_len))
		#print('longest file: ' + str(max_len))
		self.lib_gen(self.v)#necessary to call wrap() and timeline()
		print('instrument dictionary' +  str(self.inst_lib))

		return

	def generate_wv(self):

		for i, a in enumerate(self.v):
			self.wv.append(self.wrap(a))

		#print(str(self.wv[0]))

		return


	def generate_twv(self):

		for i, a in enumerate(self.wv):
			self.twv.append(self.timeline(a))

		#print(str(self.twv[0]))

		return


if __name__ == "__main__":

	q = input_block(True,0,0,0,0)
	#q.quantize_folder()
	q.generate_v()
	q.generate_wv()
	q.generate_twv()