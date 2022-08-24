"""
	Definition of several array geometries and the AcousticScene class to store everything needed to simulate the
	trajectories and the DOA estimation results.
	Pytorch datasets for sound source signals and for AcousticScenes with random trajectories and with the LOCATA
	dataset recordings.

	File name: acousticTrackingDataset.py
	Author: David Diaz-Guerra
	Date creation: 05/2020
	Python Version: 3.8
	Pytorch Version: 1.4.0
"""

import numpy as np
import os
from collections import namedtuple
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.nn import functional as F
import scipy
import scipy.io.wavfile
import soundfile
import pandas
import warnings
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import webrtcvad

from utils import rms_angular_error_deg

import gpuRIR
import math



# %% Util functions

def acoustic_power(s):
	""" Acoustic power of after removing the silences.
	"""
	w = 512  # Window size for silent detection
	o = 256  # Window step for silent detection

	# Window the input signal
	s = np.ascontiguousarray(s)
	sh = (s.size - w + 1, w)
	st = s.strides * 2
	S = np.lib.stride_tricks.as_strided(s, strides=st, shape=sh)[0::o]

	window_power = np.mean(S ** 2, axis=-1)
	th = 0.01 * window_power.max()  # Threshold for silent detection
	return np.mean(window_power[np.nonzero(window_power > th)])


def cart2sph(cart):
	xy2 = cart[:,0]**2 + cart[:,1]**2
	sph = np.zeros_like(cart)
	sph[:,0] = np.sqrt(xy2 + cart[:,2]**2)
	sph[:,1] = np.arctan2(np.sqrt(xy2), cart[:,2]) # Elevation angle defined from Z-axis down
	sph[:,2] = np.arctan2(cart[:,1], cart[:,0])
	return sph


# %% Util classes

class Parameter:
	""" Random parammeter class.
	You can indicate a constant value or a random range in its constructor and then
	get a value acording to that with getValue(). It works with both scalars and vectors.
	"""
	def __init__(self, *args):
		if len(args) == 1:
			self.random = False
			self.value = np.array(args[0])
			self.min_value = None
			self.max_value = None
		elif len(args) == 2:
			self. random = True
			self.min_value = np.array(args[0])
			self.max_value = np.array(args[1])
			self.value = None
		else: 
			raise Exception('Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
	
	def getValue(self):
		if self.random:
			return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
		else:
			return self.value


# Named tuple with the characteristics of a microphone array and definitions of the LOCATA arrays:
ArraySetup = namedtuple('ArraySetup', 'arrayType, orV, mic_pos, mic_orV, mic_pattern')

dicit_array_setup = ArraySetup(arrayType='planar', 
	orV = np.array([0.0, 1.0, 0.0]),
	mic_pos = np.array((( 0.96, 0.00, 0.00),
						( 0.64, 0.00, 0.00),
						( 0.32, 0.00, 0.00),
						( 0.16, 0.00, 0.00),
						( 0.08, 0.00, 0.00),
						( 0.04, 0.00, 0.00),
						( 0.00, 0.00, 0.00),
						( 0.96, 0.00, 0.32),
						(-0.04, 0.00, 0.00),
						(-0.08, 0.00, 0.00),
						(-0.16, 0.00, 0.00),
						(-0.32, 0.00, 0.00),
						(-0.64, 0.00, 0.00),
						(-0.96, 0.00, 0.00),
						(-0.96, 0.00, 0.32))), 
	mic_orV = np.tile(np.array([[0.0, 1.0, 0.0]]), (15,1)),
	mic_pattern = 'omni'
)

dummy_array_setup = ArraySetup(arrayType='planar', 
	orV = np.array([0.0, 1.0, 0.0]),
	mic_pos = np.array(((-0.079,  0.000, 0.000),
						(-0.079, -0.009, 0.000),
						( 0.079,  0.000, 0.000),
						( 0.079, -0.009, 0.000))), 
	mic_orV = np.array(((-1.0, 0.0, 0.0),
						(-1.0, 0.0, 0.0),
						( 1.0, 0.0, 0.0),
						( 1.0, 0.0, 0.0))), 
	mic_pattern = 'omni'
)

benchmark2_array_setup = ArraySetup(arrayType='3D', 
	orV = np.array([0.0, 1.0, 0.0]),
	mic_pos = np.array(((-0.028,  0.030, -0.040),
						( 0.006,  0.057,  0.000),
						( 0.022,  0.022, -0.046),
						(-0.055, -0.024, -0.025),
						(-0.031,  0.023,  0.042),
						(-0.032,  0.011,  0.046),
						(-0.025, -0.003,  0.051),
						(-0.036, -0.027,  0.038),
						(-0.035, -0.043,  0.025),
						( 0.029, -0.048, -0.012),
						( 0.034, -0.030,  0.037),
						( 0.035,  0.025,  0.039))), 
	mic_orV = np.array(((-0.028,  0.030, -0.040),
						( 0.006,  0.057,  0.000),
						( 0.022,  0.022, -0.046),
						(-0.055, -0.024, -0.025),
						(-0.031,  0.023,  0.042),
						(-0.032,  0.011,  0.046),
						(-0.025, -0.003,  0.051),
						(-0.036, -0.027,  0.038),
						(-0.035, -0.043,  0.025),
						( 0.029, -0.048, -0.012),
						( 0.034, -0.030,  0.037),
						( 0.035,  0.025,  0.039))),
	mic_pattern = 'omni'
)

eigenmike_array_setup = ArraySetup(arrayType='3D', 
	orV = np.array([0.0, 1.0, 0.0]),
	mic_pos = np.array((( 0.000,  0.039,  0.015),
						(-0.022,  0.036,  0.000),
						( 0.000,  0.039, -0.015),
						( 0.022,  0.036,  0.000),
						( 0.000,  0.022,  0.036),
						(-0.024,  0.024,  0.024),
						(-0.039,  0.015,  0.000),
						(-0.024,  0.024,  0.024),
						( 0.000,  0.022, -0.036),
						( 0.024,  0.024, -0.024),
						( 0.039,  0.015,  0.000),
						( 0.024,  0.024,  0.024),
						(-0.015,  0.000,  0.039),
						(-0.036,  0.000,  0.022),
						(-0.036,  0.000, -0.022),
						(-0.015,  0.000, -0.039),
						( 0.000, -0.039,  0.015),
						( 0.022, -0.036,  0.000),
						( 0.000, -0.039, -0.015),
						(-0.022, -0.036,  0.000),
						( 0.000, -0.022,  0.036),
						( 0.024, -0.024,  0.024),
						( 0.039, -0.015,  0.000),
						( 0.024, -0.024, -0.024),
						( 0.000, -0.022, -0.036),
						(-0.024, -0.024, -0.024),
						(-0.039, -0.015,  0.000),
						(-0.024, -0.024,  0.024),
						( 0.015,  0.000,  0.039),
						( 0.036,  0.000,  0.022),
						( 0.036,  0.000, -0.022),
						( 0.015,  0.000, -0.039))), 
	mic_orV = np.array((( 0.000,  0.039,  0.015),
						(-0.022,  0.036,  0.000),
						( 0.000,  0.039, -0.015),
						( 0.022,  0.036,  0.000),
						( 0.000,  0.022,  0.036),
						(-0.024,  0.024,  0.024),
						(-0.039,  0.015,  0.000),
						(-0.024,  0.024,  0.024),
						( 0.000,  0.022, -0.036),
						( 0.024,  0.024, -0.024),
						( 0.039,  0.015,  0.000),
						( 0.024,  0.024,  0.024),
						(-0.015,  0.000,  0.039),
						(-0.036,  0.000,  0.022),
						(-0.036,  0.000, -0.022),
						(-0.015,  0.000, -0.039),
						( 0.000, -0.039,  0.015),
						( 0.022, -0.036,  0.000),
						( 0.000, -0.039, -0.015),
						(-0.022, -0.036,  0.000),
						( 0.000, -0.022,  0.036),
						( 0.024, -0.024,  0.024),
						( 0.039, -0.015,  0.000),
						( 0.024, -0.024, -0.024),
						( 0.000, -0.022, -0.036),
						(-0.024, -0.024, -0.024),
						(-0.039, -0.015,  0.000),
						(-0.024, -0.024,  0.024),
						( 0.015,  0.000,  0.039),
						( 0.036,  0.000,  0.022),
						( 0.036,  0.000, -0.022),
						( 0.015,  0.000, -0.039))),
	mic_pattern = 'omni'
)

miniDSP_array_setup = ArraySetup(arrayType='planar',
	orV = np.array([0.0, 0.0, 1.0]),
	mic_pos = np.array((( 0.0000,  0.0430, 0.000),
						( 0.0372,  0.0215, 0.000),
						( 0.0372, -0.0215, 0.000),
						( 0.0000, -0.0430, 0.000),
						(-0.0372, -0.0215, 0.000),
						(-0.0372,  0.0215, 0.000))),
	mic_orV = np.array(((0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0))),
	mic_pattern = 'omni'
)


class AcousticScene:
	""" Acoustic scene class.
	It contains everything needed to simulate a moving sound source moving recorded
	with a microphone array in a reverberant room.
	It can also store the results from the DOA estimation.
	"""
	def __init__(self, room_sz, T60, beta, SNR, array_setup, mic_pos, source_signal, fs, traj_pts, timestamps,
				 trajectory, t, DOA):
		self.room_sz = room_sz				# Room size
		self.T60 = T60						# Reverberation time of the simulated room
		self.beta = beta					# Reflection coefficients of the walls of the room (make sure it corresponds with T60)
		self.SNR = SNR						# Signal to (omnidirectional) Noise Ration to simulate
		self.array_setup = array_setup		# Named tuple with the characteristics of the array
		self.mic_pos = mic_pos				# Position of the center of the array
		self.source_signal = source_signal  # Source signal
		self.fs = fs						# Sampling frequency of the source signal and the simulations
		self.traj_pts = traj_pts 			# Trajectory points to simulate
		self.timestamps = timestamps		# Time of each simulation (it does not need to correspond with the DOA estimations)
		self.trajectory = trajectory		# Continuous trajectory
		self.t = t							# Continuous time
		self.DOA = DOA 						# Continuous DOA
		self.need_direct_path_signals = True

	def simulate(self):
		""" Get the array recording using gpuRIR to perform the acoustic simulations.
		"""
		if self.T60 == 0:
			Tdiff = 0.1
			Tmax = 0.1
			nb_img = [1,1,1]
		else:
			Tdiff = gpuRIR.att2t_SabineEstimator(12, self.T60) # Use ISM until the RIRs decay 12dB
			Tmax = gpuRIR.att2t_SabineEstimator(40, self.T60)  # Use diffuse model until the RIRs decay 40dB
			if self.T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
			nb_img = gpuRIR.t2n( Tdiff, self.room_sz )

		nb_mics  = len(self.mic_pos)
		nb_traj_pts = len(self.traj_pts)
		nb_gpu_calls = min(int(np.ceil( self.fs * Tdiff * nb_mics * nb_traj_pts * np.prod(nb_img) / 1e9 )), nb_traj_pts)
		traj_pts_batch = np.ceil( nb_traj_pts / nb_gpu_calls * np.arange(0, nb_gpu_calls+1) ).astype(int)

		RIRs_list = [ gpuRIR.simulateRIR(self.room_sz, self.beta,
						 self.traj_pts[traj_pts_batch[0]:traj_pts_batch[1],:], self.mic_pos,
						 nb_img, Tmax, self.fs, Tdiff=Tdiff,
						 orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern) ]
		for i in range(1,nb_gpu_calls):
			RIRs_list += [	gpuRIR.simulateRIR(self.room_sz, self.beta,
						 	self.traj_pts[traj_pts_batch[i]:traj_pts_batch[i+1],:], self.mic_pos,
						 	nb_img, Tmax, self.fs, Tdiff=Tdiff,
						 	orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern) ]
		RIRs = np.concatenate(RIRs_list, axis=0)
		mic_signals = gpuRIR.simulateTrajectory(self.source_signal, RIRs, timestamps=self.timestamps, fs=self.fs)
		mic_signals = mic_signals[0:len(self.t),:]

		# Omnidirectional noise
		dp_RIRs = gpuRIR.simulateRIR(self.room_sz, self.beta, self.traj_pts, self.mic_pos, [1,1,1], 0.1, self.fs,
									orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern)
		dp_signals = gpuRIR.simulateTrajectory(self.source_signal, dp_RIRs, timestamps=self.timestamps, fs=self.fs)
		dp_signals = dp_signals[:mic_signals.shape[0],:]
		ac_pow = np.mean([acoustic_power(dp_signals[:,i]) for i in range(dp_signals.shape[1])])
		noise = np.sqrt(ac_pow/10**(self.SNR/10)) * np.random.standard_normal(mic_signals.shape)
		mic_signals += noise

		# Apply the propagation delay to the VAD information if it exists
		if hasattr(self, 'source_vad'):
			vad = gpuRIR.simulateTrajectory(self.source_vad, dp_RIRs, timestamps=self.timestamps, fs=self.fs)
			self.vad = vad[0:len(self.t),:].mean(axis=1) > vad[0:len(self.t),:].max()*1e-3

		return mic_signals #if self.need_direct_path_signals else mic_signals

	def get_rmsae(self, exclude_silences=False):
		""" Returns the Root Mean Square Angular Error (degrees) of the DOA estimation.
		The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
		"""
		if not exclude_silences:
			return rms_angular_error_deg(torch.from_numpy(self.DOAw_pred).double(),
										 torch.from_numpy(self.DOAw).double() )
		else:
			silences = self.vad.mean(axis=1) < 2/3
			DOAw_pred = torch.from_numpy(self.DOAw_pred[np.invert(silences), :]).double()
			self.DOAw_pred[silences, :] = np.NaN
			DOAw = torch.from_numpy(self.DOAw[np.invert(silences), :]).double()
			return rms_angular_error_deg(DOAw_pred, DOAw)

	def findMapMaximum(self, exclude_silences=False):
		""" Generates the field DOAw_est_max with the DOA estimation using the SRP-PHAT maximums
		and returns its RMSAE (in degrees) if the field DOAw exists with the DOA groundtruth.
		The scene need to have the field maps with the SRP-PHAT map of each window.
		You can choose whether to include the silent frames into the RMSAE computation or not.
		"""
		max_flat_idx = self.maps.reshape((self.maps.shape[0], -1)).argmax(1)
		theta_max_idx, phi_max_idx = np.unravel_index(max_flat_idx, self.maps.shape[1:])

		# Index to angle (radians)
		if self.array_setup.arrayType == 'planar':
			theta = np.linspace(0, np.pi/2, self.maps.shape[1])
		else:
			theta= np.linspace(0, np.pi, self.maps.shape[1])
		phi = np.linspace(-np.pi, np.pi, self.maps.shape[2]+1)
		phi = phi[:-1]
		DOAw_srpMax = np.stack((theta[theta_max_idx], phi[phi_max_idx]), axis=-1)
		self.DOAw_srpMax = DOAw_srpMax

		if not exclude_silences:
			if hasattr(self, 'DOAw'):
				return rms_angular_error_deg(torch.from_numpy(self.DOAw_srpMax),
														torch.from_numpy(self.DOAw))
		else:
			silences = self.vad.mean(axis=1) < 2/3
			self.DOAw_srpMax[silences] = np.NaN
			if hasattr(self, 'DOAw'):
				return rms_angular_error_deg(torch.from_numpy(DOAw_srpMax[np.invert(silences), :]),
														 torch.from_numpy(self.DOAw[np.invert(silences), :]) )

	def plotScene(self, view='3D'):
		""" Plots the source trajectory and the microphones within the room
		"""
		assert view in ['3D', 'XYZ', 'XY', 'XZ', 'YZ']

		fig = plt.figure()

		if view == '3D' or view == 'XYZ':
			ax = Axes3D(fig)
			ax.set_xlim3d(0, self.room_sz[0])
			ax.set_ylim3d(0, self.room_sz[1])
			ax.set_zlim3d(0, self.room_sz[2])

			ax.scatter(self.traj_pts[:,0], self.traj_pts[:,1], self.traj_pts[:,2])
			ax.scatter(self.mic_pos[:,0], self.mic_pos[:,1], self.mic_pos[:,2])
			ax.text(self.traj_pts[0,0], self.traj_pts[0,1], self.traj_pts[0,2], 'start')

			ax.set_title('$T_{60}$' + ' = {:.3f}s, SNR = {:.1f}dB'.format(self.T60, self.SNR))
			ax.set_xlabel('x [m]')
			ax.set_ylabel('y [m]')
			ax.set_zlabel('z [m]')

		else:
			ax = fig.add_subplot(111)
			plt.gca().set_aspect('equal', adjustable='box')

			if view == 'XY':
				ax.set_xlim(0, self.room_sz[0])
				ax.set_ylim(0, self.room_sz[1])
				ax.scatter(self.traj_pts[:,0], self.traj_pts[:,1])
				ax.scatter(self.mic_pos[:,0], self.mic_pos[:,1])
				ax.text(self.traj_pts[0,0], self.traj_pts[0,1], 'start')
				ax.legend(['Source trajectory', 'Microphone array'])
				ax.set_xlabel('x [m]')
				ax.set_ylabel('y [m]')
			elif view == 'XZ':
				ax.set_xlim(0, self.room_sz[0])
				ax.set_ylim(0, self.room_sz[2])
				ax.scatter(self.traj_pts[:,0], self.traj_pts[:,2])
				ax.scatter(self.mic_pos[:,0], self.mic_pos[:,2])
				ax.text(self.traj_pts[0,0], self.traj_pts[0,2], 'start')
				ax.legend(['Source trajectory', 'Microphone array'])
				ax.set_xlabel('x [m]')
				ax.set_ylabel('z [m]')
			elif view == 'YZ':
				ax.set_xlim(0, self.room_sz[1])
				ax.set_ylim(0, self.room_sz[2])
				ax.scatter(self.traj_pts[:,1], self.traj_pts[:,2])
				ax.scatter(self.mic_pos[:,1], self.mic_pos[:,2])
				ax.text(self.traj_pts[0,1], self.traj_pts[0,2], 'start')
				ax.legend(['Source trajectory', 'Microphone array'])
				ax.set_xlabel('y [m]')
				ax.set_ylabel('z [m]')

		plt.show()

	def plotDOA(self):
		""" Plots the groundtruth DOA
		"""
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(self.t, self.DOA * 180/np.pi)

		ax.legend(['Elevation', 'Azimuth'])
		ax.set_xlabel('time [s]')
		ax.set_ylabel('DOA [º]')

		plt.show()

	def plotEstimation(self, legned_loc='best'):
		""" Plots the DOA groundtruth and its estimation.
		The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
		If the scene has the field DOAw_srpMax with the SRP-PHAT estimation, it also plots it.
		"""
		fig = plt.figure()
		gs = fig.add_gridspec(7, 1)
		ax = fig.add_subplot(gs[0,0])
		ax.plot(self.t, self.source_signal)
		plt.xlim(self.tw[0], self.tw[-1])
		plt.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)

		ax = fig.add_subplot(gs[1:,0])
		ax.plot(self.tw, self.DOAw * 180/np.pi)
		plt.gca().set_prop_cycle(None)
		ax.plot(self.tw, self.DOAw_pred * 180/np.pi, '--')
		if hasattr(self, 'DOAw_srpMax'):
			plt.gca().set_prop_cycle(None)
			ax.plot(self.tw, self.DOAw_srpMax * 180 / np.pi, 'x', markersize=4)

		plt.legend(['Elevation', 'Azimuth'], loc=legned_loc)
		plt.xlabel('time [s]')
		plt.ylabel('DOA [º]')

		silences = self.vad.mean(axis=1) < 2/3
		silences_idx = silences.nonzero()[0]
		start, end = [], []
		for i in silences_idx:
			if not i - 1 in silences_idx:
				start.append(i)
			if not i + 1 in silences_idx:
				end.append(i)
		for s, e in zip(start, end):
			plt.axvspan((s-0.5)*self.tw[1], (e+0.5)*self.tw[1], facecolor='0.5', alpha=0.5)

		plt.xlim(self.tw[0], self.tw[-1])
		plt.show()


	def plotMap(self, w_idx):
		""" Plots the SRP-PHAT map of the window w_idx.
		If the scene has the fields DOAw, DOAw_pred, DOAw_srpMax it also plot them.
		"""
		maps = np.concatenate((self.maps, self.maps[..., 0, np.newaxis]), axis=-1)

		thetaMax = np.pi / 2 if self.array_setup.arrayType == 'planar' else np.pi
		theta = np.linspace(0, thetaMax, maps.shape[-2])
		phi = np.linspace(-np.pi, np.pi, maps.shape[-1])

		map = maps[w_idx, ...]
		DOA = self.DOAw[w_idx, ...] if hasattr(self, 'DOAw') else None
		DOA_pred = self.DOAw_pred[w_idx, ...] if hasattr(self, 'DOAw_pred') else None
		DOA_srpMax = self.DOAw_srpMax[w_idx, ...] if hasattr(self, 'DOAw_srpMax') else None

		plot_srp_map(theta, phi, map, DOA, DOA_pred, DOA_srpMax)

	def animateScene(self, fps=10, file_name=None):
		""" Creates an animation with the SRP-PHAT maps of each window.
		The scene need to have the field maps with the SRP-PHAT map of each window.
		If the scene has the fields DOAw, DOAw_pred, DOAw_srpMax it also includes them.
		"""
		maps = np.concatenate((self.maps, self.maps[..., 0, np.newaxis]), axis=-1)
		thetaMax = np.pi/2 if self.array_setup=='planar' else np.pi
		theta = np.linspace(0, thetaMax, maps.shape[-2])
		phi = np.linspace(-np.pi, np.pi, maps.shape[-1])

		DOAw = self.DOAw if hasattr(self, 'DOAw') else None
		DOAw_pred = self.DOAw_pred if hasattr(self, 'DOAw_pred') else None
		DOAw_srpMax = self.DOAw_srpMax if hasattr(self, 'DOAw_srpMax') else None

		animate_trajectory(theta, phi, maps, fps, DOAw, DOAw_pred, DOAw_srpMax, file_name)

# %% Source signal Datasets

class LibriSpeechDataset(Dataset):
	""" Dataset with random LibriSpeech utterances.
	You need to indicate the path to the root of the LibriSpeech dataset in your file system
	and the length of the utterances in seconds.
	The dataset length is equal to the number of chapters in LibriSpeech (585 for train-clean-100 subset)
	but each time you ask for dataset[idx] you get a random segment from that chapter.
	It uses webrtcvad to clean the silences from the LibriSpeech utterances.
	"""

	def _exploreCorpus(self, path, file_extension):
		directory_tree = {}
		for item in os.listdir(path):
			if os.path.isdir( os.path.join(path, item) ):
				directory_tree[item] = self._exploreCorpus( os.path.join(path, item), file_extension )
			elif item.split(".")[-1] == file_extension:
				directory_tree[ item.split(".")[0] ] = os.path.join(path, item)
		return directory_tree

	def _cleanSilences(self, s, aggressiveness, return_vad=False):
		self.vad.set_mode(aggressiveness)

		vad_out = np.zeros_like(s)
		vad_frame_len = int(10e-3 * self.fs)
		n_vad_frames = len(s) // vad_frame_len
		for frame_idx in range(n_vad_frames):
			frame = s[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
			frame_bytes = (frame * 32767).astype('int16').tobytes()
			vad_out[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = self.vad.is_speech(frame_bytes, self.fs)
		s_clean = s * vad_out

		return (s_clean, vad_out) if return_vad else s_clean

	def __init__(self, path, T, size=None, return_vad=False, readers_range=None):
		self.corpus = self._exploreCorpus(path, 'flac')
		if readers_range is not None:
			for key in list(map(int, self.nChapters.keys())):
				if int(key) < readers_range[0] or int(key) > readers_range[1]:
					del self.corpus[key]

		self.nReaders = len(self.corpus)
		self.nChapters = {reader: len(self.corpus[reader]) for reader in self.corpus.keys()}
		self.nUtterances = {reader: {
				chapter: len(self.corpus[reader][chapter]) for chapter in self.corpus[reader].keys()
			} for reader in self.corpus.keys()}

		self.chapterList = []
		for chapters in list(self.corpus.values()):
			self.chapterList += list(chapters.values())

		self.fs = 16000
		self.T = T

		self.return_vad = return_vad
		self.vad = webrtcvad.Vad()

		self.sz = len(self.chapterList) if size is None else size

	def __len__(self):
		return self.sz

	def __getitem__(self, idx):
		if idx < 0: idx = len(self) + idx
		while idx >= len(self.chapterList): idx -= len(self.chapterList)
		chapter = self.chapterList[idx]

		# Get a random speech segment from the selected chapter
		s = np.array([])
		utt_paths = list(chapter.values())
		n = np.random.randint(0,len(chapter))
		while s.shape[0] < self.T * self.fs:
			utterance, fs = soundfile.read(utt_paths[n])
			assert fs == self.fs
			s = np.concatenate([s, utterance])
			n += 1
			if n >= len(chapter): n=0
		s = s[0: self.T * fs]
		s -= s.mean()

		# Clean silences, it starts with the highest aggressiveness of webrtcvad,
		# but it reduces it if it removes more than the 66% of the samples
		s_clean, vad_out = self._cleanSilences(s, 3, return_vad=True)
		if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
			s_clean, vad_out = self._cleanSilences(s, 2, return_vad=True)
		if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
			s_clean, vad_out = self._cleanSilences(s, 1, return_vad=True)

		return (s_clean, vad_out) if self.return_vad else s_clean


# %% Acoustic Scene Datasets

class RandomTrajectoryDataset(Dataset):
	""" Dataset Acoustic Scenes with random trajectories.
	The length of the dataset is the length of the source signals dataset.
	When you access to an element you get both the simulated signals in the microphones and the AcousticScene object.
	"""
	def __init__(self, sourceDataset, room_sz, T60, abs_weights, array_setup, array_pos, SNR, nb_points, transforms=None):
		"""
		sourceDataset: dataset with the source signals (such as LibriSpeechDataset)
		room_sz: Size of the rooms in meters
		T60: Reverberation time of the room in seconds
		abs_weights: Absorption coefficients rations of the walls
		array_setup: Named tuple with the characteristics of the array
		array_pos: Position of the center of the array as a fraction of the room size
		SNR: Signal to (omnidirectional) Noise Ration
		nb_points: Number of points to simulate along the trajectory
		transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene

		Any parameter (except from nb_points and transforms) can be Parameter object to make it random.
		"""
		self.sourceDataset = sourceDataset

		self.room_sz = room_sz if type(room_sz) is Parameter else Parameter(room_sz)
		self.T60 = T60 if type(T60) is Parameter else Parameter(T60)
		self.abs_weights = abs_weights if type(abs_weights) is Parameter else Parameter(abs_weights)

		assert np.count_nonzero(array_setup.orV) == 1, "array_setup.orV mus be parallel to an axis"
		self.array_setup = array_setup
		self.N = array_setup.mic_pos.shape[0]
		self.num_mic_pairs = int((self.N*(self.N -1))//2)
		self.mic_pairs = [(_ch1, _ch2) for _ch1 in range(self.N) for _ch2 in range(_ch1+1, self.N)]  #List of Tuples (mic1, mic2)
		self.array_pos = array_pos if type(array_pos) is Parameter else Parameter(array_pos)

		self.SNR = SNR if type(SNR) is Parameter else Parameter(SNR)
		self.nb_points = nb_points
		self.fs = sourceDataset.fs

		self.transforms = transforms

	def __len__(self):
		"include each microphone pair separately"
		return len(self.sourceDataset)*self.num_mic_pairs

	def __getitem__(self, idx):
		if idx < 0: idx = len(self) + idx

		speech_file_idx = idx // len(self.sourceDataset)
		mic_pair = self.mic_pairs[ (idx % self.num_mic_pairs)]
		acoustic_scene = self.getRandomScene(speech_file_idx)

		mic_signals = acoustic_scene.simulate()
		mic_signals = torch.from_numpy(mic_signals.T)

		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, direct_path_phdiff, acoustic_scene = t(mic_signals, acoustic_scene, mic_pair)

		return mic_signals, direct_path_phdiff, acoustic_scene.doA_reduced_frms #np.transpose(acoustic_scene.DOA)

	def get_batch(self, idx1, idx2):
		mic_sig_batch = []
		acoustic_scene_batch = []
		for idx in range(idx1, idx2):
			mic_sig, acoustic_scene = self[idx]
			mic_sig_batch.append(mic_sig)
			acoustic_scene_batch.append(acoustic_scene)

		return np.stack(mic_sig_batch), np.stack(acoustic_scene_batch)

	def getRandomScene(self, idx):
		# Source signal
		source_signal, vad = self.sourceDataset[idx]

		# Room
		room_sz = self.room_sz.getValue()
		T60 = self.T60.getValue()
		abs_weights = self.abs_weights.getValue()
		beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights)

		# Microphones
		array_pos = self.array_pos.getValue() * room_sz
		mic_pos = array_pos + self.array_setup.mic_pos

		# Trajectory points
		src_pos_min = np.array([0.0, 0.0, 0.0])
		src_pos_max = room_sz
		if self.array_setup.arrayType == 'planar':
			if np.sum(self.array_setup.orV) > 0:
				src_pos_min[np.nonzero(self.array_setup.orV)] = array_pos[np.nonzero(self.array_setup.orV)]
			else:
				src_pos_max[np.nonzero(self.array_setup.orV)] = array_pos[np.nonzero(self.array_setup.orV)]
		src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
		src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)

		Amax = np.min(np.stack((src_pos_ini - src_pos_min,
									  src_pos_max - src_pos_ini,
									  src_pos_end - src_pos_min,
									  src_pos_max - src_pos_end)),
								axis=0)

		A = np.random.random(3) * np.minimum(Amax, 1) 			# Oscilations with 1m as maximum in each axis
		w = 2*np.pi / self.nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis

		traj_pts = np.array([np.linspace(i,j,self.nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
		traj_pts += A * np.sin(w * np.arange(self.nb_points)[:, np.newaxis])

		if np.random.random(1) < 0.25:
			traj_pts = np.ones((self.nb_points,1)) * src_pos_ini

		# Interpolate trajectory points
		timestamps = np.arange(self.nb_points) * len(source_signal) / self.fs / self.nb_points
		t = np.arange(len(source_signal))/self.fs
		trajectory = np.array([np.interp(t, timestamps, traj_pts[:,i]) for i in range(3)]).transpose()

		acoustic_scene = AcousticScene(
			room_sz = room_sz,
			T60 = T60,
			beta = beta,
			SNR = self.SNR.getValue(),
			array_setup = self.array_setup,
			mic_pos = mic_pos,
			source_signal = source_signal,
			fs = self.fs,
			t = t,
			traj_pts = traj_pts,
			timestamps = timestamps,
			trajectory = trajectory,
			DOA = cart2sph(trajectory - array_pos)[:,:2]#[:,1:3]
		)
		acoustic_scene.source_vad = vad

		return acoustic_scene


class  LocataDataset(Dataset):
	""" Dataset with the LOCATA dataset recordings and its corresponding Acoustic Scenes.
	When you access to an element you get both the simulated signals in the microphones and the AcousticScene object.
	"""
	def __init__(self, path, array, fs, tasks=(1,3,5), recording=None, dev=False, transforms = None):
		"""
		path: path to the root of the LOCATA dataset in your file system
		array: string with the desired array ('dummy', 'eigenmike', 'benchmark2' or 'dicit')
		fs: sampling frequency (you can use it to downsample the LOCATA recordings)
		tasks: LOCATA tasks to include in the dataset (only one-source tasks are supported)
		recording: recordings that you want to include in the dataset (only supported if you selected only one task)
		dev: True if the groundtruth source positions are available
		transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
		"""
		assert array in ('dummy', 'eigenmike', 'benchmark2', 'dicit'), 'Invalid array.'
		assert recording is None or len(tasks) == 1, 'Specific recordings can only be selected for dataset with only one task'
		for task in tasks: assert task in (1,3,5), 'Invalid task ' + str(task) + '.'

		self.path = path
		self.dev = dev
		self.array = array
		self.tasks = tasks
		self.transforms = transforms
		self.fs = fs

		self.vad = webrtcvad.Vad()
		self.vad.set_mode(1)

		if array == 'dummy':
			self.array_setup = dummy_array_setup
		elif array == 'eigenmike':
			self.array_setup = eigenmike_array_setup
		elif array == 'benchmark2':
			self.array_setup = benchmark2_array_setup
		elif array == 'dicit':
			self.array_setup = dicit_array_setup

		self.directories = []
		for task in tasks:
			task_path = os.path.join(path, 'task' + str(task))
			for recording in os.listdir( task_path ):
				arrays = os.listdir( os.path.join(task_path, recording) )
				if array in arrays:
					self.directories.append( os.path.join(task_path, recording, array) )
		self.directories.sort()

	def __len__(self):
		return len(self.directories)

	def __getitem__(self, idx):
		directory = self.directories[idx]
		mic_signals, fs = soundfile.read( os.path.join(directory, 'audio_array_' + self.array + '.wav') )
		if fs > self.fs:
			mic_signals = scipy.signal.decimate(mic_signals, int(fs/self.fs), axis=0)
			new_fs = fs / int(fs/self.fs)
			if new_fs != self.fs: warnings.warn('The actual fs is {}Hz'.format(new_fs))
			self.fs = new_fs
		elif fs < self.fs:
			raise Exception('The sampling rate of the file ({}Hz) was lower than self.fs ({}Hz'.format(fs, self.fs))

		# Remove initial silence
		start = np.argmax(mic_signals[:,0] > mic_signals[:,0].max()*0.15)
		mic_signals = mic_signals[start:,:]
		t = (np.arange(len(mic_signals)) + start)/self.fs

		df = pandas.read_csv( os.path.join(directory, 'position_array_' + self.array + '.txt'), sep='\t' )
		array_pos = np.stack((df['x'].values, df['y'].values,df['z'].values), axis=-1)
		array_ref_vec = np.stack((df['ref_vec_x'].values, df['ref_vec_y'].values,df['ref_vec_z'].values), axis=-1)
		array_rotation = np.zeros((array_pos.shape[0],3,3))
		for i in range(3):
			for j in range(3):
				array_rotation[:,i,j] = df['rotation_' + str(i+1) + str(j+1)]

		df = pandas.read_csv( os.path.join(directory, 'required_time.txt'), sep='\t' )
		required_time = df['hour'].values*3600+df['minute'].values*60+df['second'].values
		timestamps = required_time - required_time[0]

		if self.dev:
			sources_pos = []
			trajectories = []
			for file in os.listdir( directory ):
				if file.startswith('audio_source') and file.endswith('.wav'):
					source_signal, fs_src = soundfile.read(os.path.join(directory, file))
					if fs > self.fs:
						source_signal = scipy.signal.decimate(source_signal, int(fs_src / self.fs), axis=0)
					source_signal = source_signal[start:start+len(t)]
				if file.startswith('position_source'):
					df = pandas.read_csv( os.path.join(directory, file), sep='\t' )
					source_pos = np.stack((df['x'].values, df['y'].values,df['z'].values), axis=-1)
					sources_pos.append( source_pos )
					trajectories.append( np.array([np.interp(t, timestamps, source_pos[:,i]) for i in range(3)]).transpose() )
			sources_pos = np.stack(sources_pos)
			trajectories = np.stack(trajectories)

			DOA_pts = np.zeros(sources_pos.shape[0:2] + (2,))
			DOA = np.zeros(trajectories.shape[0:2] + (2,))
			for s in range(sources_pos.shape[0]):
				source_pos_local = np.matmul( np.expand_dims(sources_pos[s,...] - array_pos, axis=1), array_rotation ).squeeze() # np.matmul( array_rotation, np.expand_dims(sources_pos[s,...] - array_pos, axis=-1) ).squeeze()
				DOA_pts[s,...] = cart2sph(source_pos_local) [:,1:3]
				DOA[s,...] = np.array([np.interp(t, timestamps, DOA_pts[s,:,i]) for i in range(2)]).transpose()
			DOA[DOA[...,1]<-np.pi, 1] += 2*np.pi
		else:
			sources_pos = None
			DOA = None
			source_signal = np.NaN * np.ones((len(mic_signals),1))

		acoustic_scene = AcousticScene(
			room_sz = np.NaN * np.ones((3,1)),
			T60 = np.NaN,
			beta = np.NaN * np.ones((6,1)),
			SNR = np.NaN,
			array_setup = self.array_setup,
			mic_pos = np.matmul( array_rotation[0,...], np.expand_dims(self.array_setup.mic_pos, axis=-1) ).squeeze() + array_pos[0,:], # self.array_setup.mic_pos + array_pos[0,:], # Not valid for moving arrays
			source_signal = source_signal,
			fs = self.fs,
			t = t - start/self.fs,
			traj_pts = sources_pos[0,...],
			timestamps = timestamps - start/self.fs,
			trajectory = trajectories[0,...],
			DOA = DOA[0,...]
		)

		vad = np.zeros_like(source_signal)
		vad_frame_len = int(10e-3 * self.fs)
		n_vad_frames = len(source_signal) // vad_frame_len
		for frame_idx in range(n_vad_frames):
			frame = source_signal[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
			frame_bytes = (frame * 32767).astype('int16').tobytes()
			vad[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = self.vad.is_speech(frame_bytes, int(self.fs))
		acoustic_scene.vad = vad

		#mic_signals.transpose()
		print(f"{mic_signals.shape}")
		mic_signals = torch.from_numpy(mic_signals.T.copy())

		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, direct_path_phdiff, acoustic_scene = t(mic_signals, acoustic_scene)

		return mic_signals, direct_path_phdiff, acoustic_scene.doA_reduced_frms

	def get_batch(self, idx1, idx2):
		mic_sig_batch = []
		acoustic_scene_batch = []
		for idx in range(idx1, idx2):
			mic_sig, acoustic_scene = self[idx]
			mic_sig_batch.append(mic_sig)
			acoustic_scene_batch.append(acoustic_scene)

		return np.stack(mic_sig_batch), np.stack(acoustic_scene_batch)


# %% Transform classes

class Windowing(object):
	""" Windowing transform.
	Create it indicating the window length (K), the step between windows and an optional
	window shape indicated as a vector of length K or as a Numpy window function.
	"""
	def __init__(self, K, step, window=None):
		self.K = K
		self.step = step
		if window is None:
			self.w = np.ones(K)
		elif callable(window):
			try: self.w = window(K)
			except: raise Exception('window must be a NumPy window function or a Numpy vector with length K')
		elif len(window) == K:
			self.w = window
		else:
			raise Exception('window must be a NumPy window function or a Numpy vector with length K')

	def __call__(self, x, acoustic_scene):
		N_mics = x.shape[1]
		N_dims = acoustic_scene.DOA.shape[1]
		L = x.shape[0]
		N_w = np.floor(L/self.step - self.K/self.step + 1).astype(int)

		if self.K > L:
			raise Exception('The window size can not be larger than the signal length ({})'.format(L))
		elif self.step > L:
			raise Exception('The window step can not be larger than the signal length ({})'.format(L))

		# Pad the signal and the DOA
		x = np.append(x, np.zeros((N_w * self.step + self.K - L, N_mics)), axis=0)
		DOA = np.append(acoustic_scene.DOA,
						np.tile(acoustic_scene.DOA[-1,:].reshape((1,2)), [N_w*self.step+self.K-L, 1]),
						axis=0) # Replicate the last known DOA
		L = x.shape[0]

		# Window the signal
		shape_Xw = (N_w, self.K, N_mics)
		strides_Xw = [self.step*N_mics, N_mics, 1]
		strides_Xw = [strides_Xw[i] * x.itemsize for i in range(3)]
		Xw = np.lib.stride_tricks.as_strided(x, shape=shape_Xw, strides=strides_Xw)
		Xw = Xw.transpose((0,2,1)) * self.w

		# Window the DOA
		shape_DOAw = (N_w, self.K, N_dims)
		strides_DOAw = [self.step*N_dims, N_dims, 1]
		strides_DOAw = [strides_DOAw[i] * DOA.itemsize for i in range(3)]
		DOAw = np.lib.stride_tricks.as_strided(DOA, shape=shape_DOAw, strides=strides_DOAw)
		DOAw = np.ascontiguousarray(DOAw)
		for i in np.flatnonzero(np.abs(np.diff(DOAw[..., 1], axis=1)).max(axis=1) > np.pi):
			DOAw[i,  DOAw[i,:,1]<0, 1] += 2*np.pi # Avoid jumping from -pi to pi in a window
		DOAw = np.mean(DOAw, axis=1)
		DOAw[DOAw[:,1]>np.pi, 1] -= 2*np.pi
		acoustic_scene.DOAw = DOAw

		# Window the VAD if it exists
		if hasattr(acoustic_scene, 'vad'):
			vad = acoustic_scene.vad[:, np.newaxis]
			vad = np.append(vad, np.zeros((L - vad.shape[0], 1)), axis=0)

			shape_vadw = (N_w, self.K, 1)
			strides_vadw = [self.step * 1, 1, 1]
			strides_vadw = [strides_vadw[i] * vad.itemsize for i in range(3)]

			acoustic_scene.vad = np.lib.stride_tricks.as_strided(vad, shape=shape_vadw, strides=strides_vadw)[...,0]

		# Timestamp for each window
		acoustic_scene.tw = np.arange(0, (L-self.K), self.step) / acoustic_scene.fs

		return Xw, acoustic_scene


class Extract_DOAw(object):
	""" Replace the AcousticScene object by just its windowed DOA
	"""
	def __call__(self, x, acoustic_scene):
		return x, acoustic_scene.DOAw


class ToFloat32(object):
	""" Convert to np.float32
	"""
	def __call__(self, x, DOAw):
		return x.astype(np.float32), DOAw.astype(np.float32)


# %% Representation functions

def plot_srp_map(theta, phi, srp_map, DOA=None, DOA_est=None, DOA_srpMax=None):
	theta = theta * 180/np.pi
	phi = phi * 180/np.pi
	theta_step = theta[1] - theta[0]
	phi_step = phi[1] - phi[0]
	plt.imshow(srp_map, cmap='inferno', extent=(phi[0]-phi_step/2, phi[-1]+phi_step/2, theta[-1]+theta_step/2, theta[0]-theta_step/2))
	plt.xlabel('Azimuth [º]')
	plt.ylabel('Elevation [º]')

	if DOA is not None:
		if DOA.ndim == 1: plt.scatter(DOA[1]*180/np.pi, DOA[0]*180/np.pi, c='r')
		elif DOA.ndim == 2:
			DOA_s = np.split(DOA, (np.abs(np.diff(DOA[:, 1])) > np.pi).nonzero()[0] + 1) # Split when jumping from -pi to pi
			[plt.plot(DOA_s[i][:, 1]*180/np.pi, DOA_s[i][:, 0]*180/np.pi, 'r') for i in range(len(DOA_s))]
			plt.scatter(DOA[-1,1]*180/np.pi, DOA[-1,0]*180 / np.pi, c='r')
	if DOA_srpMax is not None:
		if DOA_srpMax.ndim == 1: plt.scatter(DOA_srpMax[1] *180/np.pi, DOA_srpMax[0]*180/np.pi, c='k')
		elif DOA_srpMax.ndim == 2:
			DOA_srpMax_s = np.split(DOA_srpMax, (np.abs(np.diff(DOA_srpMax[:, 1])) > np.pi).nonzero()[0] + 1) # Split when jumping from -pi to pi
			[plt.plot(DOA_srpMax_s[i][:, 1]*180 / np.pi, DOA_srpMax_s[i][:, 0]*180 / np.pi, 'k') for i in range(len(DOA_srpMax_s))]
			plt.scatter(DOA_srpMax[-1,1]*180 / np.pi, DOA_srpMax[-1,0]*180 / np.pi, c='k')
	if DOA_est is not None:
		if DOA_est.ndim == 1: plt.scatter(DOA_est[1]*180/np.pi, DOA_est[0]*180/np.pi, c='b')
		elif DOA_est.ndim == 2:
			DOA_est_s = np.split(DOA_est, (np.abs(np.diff(DOA_est[:, 1])) > np.pi).nonzero()[0] + 1) # Split when jumping from -pi to pi
			[plt.plot(DOA_est_s[i][:, 1]*180 / np.pi, DOA_est_s[i][:, 0]*180 / np.pi, 'b') for i in range(len(DOA_est_s))]
			plt.scatter(DOA_est[-1,1]*180 / np.pi, DOA_est[-1,0]*180 / np.pi, c='b')

	plt.xlim(phi.min(), phi.max())
	plt.ylim(theta.max(), theta.min())
	plt.show()


def animate_trajectory(theta, phi, srp_maps, fps, DOA=None, DOA_est=None, DOA_srpMax=None, file_name=None):
	fig = plt.figure()

	def animation_function(frame, theta, phi, srp_maps, DOA=None, DOA_est=None, DOA_srpMax=None):
		plt.clf()
		srp_map = srp_maps[frame,:,:]
		if DOA is not None: DOA = DOA[:frame+1,:]
		if DOA_est is not None: DOA_est = DOA_est[:frame+1,:]
		if DOA_srpMax is not None: DOA_srpMax = DOA_srpMax[:frame+1,:]
		plot_srp_map(theta, phi, srp_map, DOA, DOA_est, DOA_srpMax)

	anim = animation.FuncAnimation(fig, animation_function, frames=srp_maps.shape[0], fargs=(theta, phi, srp_maps, DOA, DOA_est, DOA_srpMax), interval=1e3/fps, repeat=False)
	plt.show()
	plt.close(fig)
	if file_name is not None: anim.save(file_name, fps=fps, extra_args=['-vcodec', 'libx264'])


class spectrum_analysis(object):
	def __init__(self):
		self.window_len = 512
		self.hop_size = 256 #64
		self.window_callback = torch.hamming_window
		self.hamming_window = torch.hamming_window(self.window_len)
		self.sr = 16000

class SRPDNN_features(object):
	def __init__(self):

		super().__init__()
		self._spectrum_config = spectrum_analysis()
		self.windowing = TorchChunking(3072,3072)  #16ms *12 frms* 16kHz 

	def __call__(self, mix_signal, acoustic_scene, mic_pair):
		epsilon = 10**-10
		mix_signal = torch.cat((mix_signal[[mic_pair[0]]], mix_signal[[mic_pair[1]]]), dim=0)

		#print(f"{mix_signal.shape} {direct_path_signal.shape}")
		
		hamming_window = torch.hamming_window(self._spectrum_config.window_len)#.type_as(masked_mag)
		mix_cs = torch.stft(mix_signal,  self._spectrum_config.window_len, self._spectrum_config.hop_size, self._spectrum_config.window_len, hamming_window, return_complex=True)	
		mix_cs = mix_cs[:,1:,:] # removing the dc component

		log_ms = torch.log(torch.abs(mix_cs) + epsilon)
		mix_ph = torch.angle(mix_cs)
		log_ms_ph = torch.cat((log_ms, mix_ph), dim=0)

		mic_pos = acoustic_scene.mic_pos
		doA = np.transpose(acoustic_scene.DOA)

		doA_frm = self.windowing.get_my_frames(torch.from_numpy(doA))
		doA_reduced_frms = torch.mode(doA_frm)[0]
		acoustic_scene.doA_reduced_frms = doA_reduced_frms.to(dtype = torch.float32)

		#print(f"doA : {doA.shape}, doA_reduced_frms: {doA_reduced_frms.shape}")
		u_theta = torch.cat( ( torch.cat( ( torch.sin(doA_reduced_frms[:,[0]])*torch.cos(doA_reduced_frms[:,[1]]), torch.sin(doA_reduced_frms[:,[0]])*torch.sin(doA_reduced_frms[:,[1]]) ), dim=1), torch.cos(doA_reduced_frms[:,[0]]) ), dim = 1)
		pi = math.pi
		w = torch.from_numpy(np.array([(2*pi/512)*k for k in range(1,257)])).unsqueeze(dim=0)
		#print(f"w_shape: {w.shape}, u_theta: {u_theta.shape}")
		c = 343 

		dist_mics = torch.from_numpy(mic_pos[mic_pair[0]] - mic_pos[mic_pair[1]]).unsqueeze(dim=1)
		tou = torch.matmul(u_theta, dist_mics) / c
		#print(f"tou_shape: {tou.shape}, dist_mics: {dist_mics.shape}")
		dp_phase_diff = torch.cat((torch.cos(torch.matmul(tou, w)), torch.sin(torch.matmul(tou, w))), dim=1)
		return log_ms_ph.to(dtype=torch.float32) , dp_phase_diff.to(dtype=torch.float32), acoustic_scene  # (4,F,T), (T,2*F)

	def __callv1__(self, mix_signal, direct_path_signal, acoustic_scene):  # direct_path_signal,
		epsilon = 10**-10
		mix_cs = torch.stft(mix_signal,  self._spectrum_config.window_len, self._spectrum_config.hop_size, self._spectrum_config.window_len, self._spectrum_config.hamming_window, return_complex=True)
		#dp_cs = torch.stft(direct_path_signal,  self._spectrum_config.window_len, self._spectrum_config.hop_size, self._spectrum_config.window_len, self._spectrum_config.hamming_window, return_complex=True)
		# removing the dc component
		mix_cs = mix_cs[:,1:,:]
		#dp_cs = dp_cs[:,1:,:]

		log_ms = torch.log(torch.abs(mix_cs) + epsilon)
		mix_ph = torch.angle(mix_cs)
		#dp_phase = torch.angle(dp_cs)
		
		num_channels, _, _ = mix_cs.shape

		_log_ms_ph, _dp_phasediff = [], []

		mic_pos = acoustic_scene.mic_pos
		doA = np.transpose(acoustic_scene.DOA)

		doA_frm = self.windowing.get_my_frames(torch.from_numpy(doA))
		doA_reduced_frms = torch.mode(doA_frm)[0]
		#print(f"doA : {doA.shape}, doA_reduced_frms: {doA_reduced_frms.shape}")
		u_theta = torch.cat( ( torch.cat( ( torch.sin(doA_reduced_frms[:,[0]])*torch.cos(doA_reduced_frms[:,[1]]), torch.sin(doA_reduced_frms[:,[0]])*torch.sin(doA_reduced_frms[:,[1]]) ), dim=1), torch.cos(doA_reduced_frms[:,[0]]) ), dim = 1)
		pi = math.pi
		w = torch.from_numpy(np.array([(2*pi/512)*k for k in range(1,257)])).unsqueeze(dim=0)
		#print(f"w_shape: {w.shape}, u_theta: {u_theta.shape}")
		c = 343 
		
		for _ch1 in range(num_channels):
			for _ch2 in range(_ch1+1, num_channels):			
				dist_mics = torch.from_numpy(mic_pos[_ch1] - mic_pos[_ch2]).unsqueeze(dim=1)
				tou = torch.matmul(u_theta, dist_mics) / c
				#print(f"tou_shape: {tou.shape}, dist_mics: {dist_mics.shape}")
				out_lbl = torch.cat((torch.cos(torch.matmul(tou, w)), torch.sin(torch.matmul(tou, w))), dim=1)
				#print(f"out_lbl_shape: {out_lbl.shape}")
				_dp_phasediff.append(out_lbl.numpy()) #t()

				"""
				dp_phasediff = dp_phase[_ch1] - dp_phase[_ch2]
		
				ph_real = torch.cos(dp_phasediff)
				ph_imag = torch.sin(dp_phasediff)

				_dp_phasediff.append(torch.cat((ph_real, ph_imag), dim = 0).numpy())
				"""

				_log_ms_ph.append(torch.cat((log_ms[[_ch1]], log_ms[[_ch2]], mix_ph[[_ch1]], mix_ph[[_ch2]]), dim=0).numpy())
			
		
		acoustic_scene.doA_reduced_frms = doA_reduced_frms.to(dtype = torch.float32)

		return torch.from_numpy(np.array(_log_ms_ph)).to(dtype=torch.float32), torch.from_numpy(np.array(_dp_phasediff)).to(dtype=torch.float32), acoustic_scene  # num_channels_C_2,4,F,T :   num_channels_C_2,2*F,T
#DoA is still float64


class SRPDNN_Test_features(object):
	def __init__(self,mic1, mic2):

		super().__init__()
		self._spectrum_config = spectrum_analysis()
		self._ch1 = mic1
		self._ch2 = mic2
		self.windowing = TorchChunking(3072,3072)  #16ms *12 frms* 16kHz 

		elevations = torch.deg2rad(torch.linspace(0,180,37))  #radians
		azimuths = torch.deg2rad(torch.linspace(0,360,73))

		theta_grid = torch.cartesian_prod(elevations, azimuths)
		self.u_theta = torch.cat( ( torch.cat( ( torch.sin(theta_grid[:,[0]])*torch.cos(theta_grid[:,[1]]), torch.sin(theta_grid[:,[0]])*torch.sin(theta_grid[:,[1]]) ), dim=1), torch.cos(theta_grid[:,[0]]) ), dim = 1)
		pi = math.pi
		self.w = torch.from_numpy(np.array([(2*pi/512)*k for k in range(1,257)])).unsqueeze(dim=0).to(torch.float32)
		#print(f"w_shape: {w.shape}, u_theta: {u_theta.shape}")
		

	def __call__(self, mix_signal, acoustic_scene):  # direct_path_signal,
		epsilon = 10**-10
		mix_cs = torch.stft(mix_signal,  self._spectrum_config.window_len, self._spectrum_config.hop_size, self._spectrum_config.window_len, self._spectrum_config.hamming_window, return_complex=True)		
		mix_cs = mix_cs[:,1:,:] # removing the dc component

		log_ms = torch.log(torch.abs(mix_cs) + epsilon)
		mix_ph = torch.angle(mix_cs)

		num_channels, _, _ = mix_cs.shape

		_log_ms_ph, _dp_phasediff = [], []

		mic_pos = acoustic_scene.mic_pos
		doA = np.transpose(acoustic_scene.DOA)

		doA_frm = self.windowing.get_my_frames(torch.from_numpy(doA))
		doA_reduced_frms = torch.mode(doA_frm)[0]
		c = 343 
		for _ch1 in range(num_channels):
			for _ch2 in range(_ch1+1, num_channels):			
				dist_mics = torch.from_numpy(mic_pos[_ch1] - mic_pos[_ch2]).unsqueeze(dim=1)
				dist_mics = dist_mics.to(dtype = torch.float32) # torch.transpose(, 0, 1)
				tou = torch.matmul(self.u_theta, dist_mics) / c
				#print(f"tou_shape: {tou.shape}, dist_mics: {dist_mics.shape}")
				out_lbl = torch.cat((torch.cos(torch.matmul(tou, self.w)), torch.sin(torch.matmul(tou, self.w))), dim=1)
				#print(f"out_lbl_shape: {out_lbl.shape}")
				_dp_phasediff.append(out_lbl.numpy()) #t()

				_log_ms_ph.append(torch.cat((log_ms[[_ch1]], log_ms[[_ch2]], mix_ph[[_ch1]], mix_ph[[_ch2]]), dim=0).numpy())
				
		acoustic_scene.doA_reduced_frms = doA_reduced_frms.to(dtype = torch.float32)

		return torch.from_numpy(np.array(_log_ms_ph)).to(dtype=torch.float32) , torch.from_numpy(np.array(_dp_phasediff)).to(dtype=torch.float32) , acoustic_scene  # num_channels_C_2,4,F,T :   num_channels_C_2,2*F,T


class TorchChunking(nn.Module):
	def __init__(self, frame_size=512, frame_shift=256, pad_left=0):
		super(TorchChunking, self).__init__()
		self.frame_size = frame_size
		self.frame_shift = frame_shift
		self.pad_left = pad_left
		self.kernel_size = (self.frame_size, 1)
		self.stride = (self.frame_shift, 1)

	def get_frames(self, in_sig):
		N = in_sig.shape[-1]
		N_new = N + self.pad_left
		pad_right = (N_new//self.frame_shift - 1) * \
			self.frame_shift + self.frame_size - N_new
		out = F.pad(in_sig, (self.pad_left, pad_right))
		out = torch.unsqueeze(out, dim=-1)
		out = F.unfold(out, kernel_size=self.kernel_size,
					   stride=self.stride,
					   padding=(0, 0))
		out = out.transpose(-2, -1)
		return out

	def ola(self, inputs, size):
		inputs = inputs.transpose(-2, -1)
		den = torch.ones_like(inputs)
		N_new = size + self.pad_left
		pad_right = (N_new//self.frame_shift-1) * \
			self.frame_shift + self.frame_size - N_new
		out = F.fold(inputs, output_size=(size+self.pad_left+pad_right, 1),
					 kernel_size=self.kernel_size,
					 stride=self.stride,
					 padding=(0, 0))

		den = F.fold(den, output_size=(size+self.pad_left+pad_right, 1),
					 kernel_size=self.kernel_size,
					 stride=self.stride,
					 padding=(0, 0))
		out = out / den
		out = out.squeeze(dim=-1)
		return out[..., self.pad_left:self.pad_left+size]

	def get_my_frames(self, x):
		size = x.shape[-1]
		y = x[:, 0:self.frame_size].unsqueeze(dim=0)
		for frm_strt in range(self.frame_shift, size - self.frame_size, self.frame_shift):
  			#print(f"{frm_strt} : {y.shape}")
  			y = torch.cat([y, x[:, frm_strt:frm_strt + self.frame_size].unsqueeze(dim=0)], dim = 0)

		return y


class paths(object):
	def __init__(self):
		self.scratch_dir = "/fs/scratch/PAS0774/Shanmukh"
		self.rel_train_file_path = "/Databases/LibriSpeech/LibriSpeech/train-clean-100"
		self.rel_dev_file_path = "/Databases/LibriSpeech/LibriSpeech/dev-clean"
		self.rel_test_file_path = "/Databases/LibriSpeech/LibriSpeech/test-clean"
		self.rel_model_path = "/MVTSS/Baselines/SRPDNN/lr_1e-4_32bit_precision/"
		self.rel_log_path = "/MVTSS/Baselines/SRPDNN/lr_1e-4_32bit_precision/"
		self.train_file_path = self.scratch_dir + self.rel_train_file_path
		self.dev_file_path = self.scratch_dir + self.rel_dev_file_path
		self.test_file_path = self.scratch_dir + self.rel_test_file_path
		self.model_path = self.scratch_dir + self.rel_model_path
		self.log_path = self.scratch_dir + self.rel_log_path
		self.model_ckpt_name = ""

		self.last_ckpt_name = "last.ckpt"    #Latest train model to resume training
		self.abs_model_file_name = self.model_path + self.model_ckpt_name
		self.abs_last_model_file_name = self.model_path + self.last_ckpt_name



if __name__=="__main__":
	_locata=False
	if not _locata:
		T = 20 # Trajectory length (s)
		_libri_speech_path = '/fs/scratch/PAS0774/Shanmukh/Databases/LibriSpeech/LibriSpeech/train-clean-100'
		array_setup = benchmark2_array_setup
		nb_points = 64
		room_sz = Parameter([3,3,2.5], [10,8,6]) 	# Random room sizes from 3x3x2.5 to 10x8x6 meters
		T60 = Parameter(0.2, 1.3)					# Random reverberation times from 0.2 to 1.3 seconds
		abs_weights = Parameter([0.5]*6, [1.0]*6)  # Random absorption weights ratios between walls
		SNR = Parameter(5, 30)
		array_pos = Parameter([0.4, 0.1, 0.1],[0.6, 0.9, 0.5]) # 

		#_val1, _val2 = array_pos.getValue(), room_sz.getValue()
		#array_pos = _val1 * _val2
		#mic_pos = array_pos + array_setup.mic_pos
		#print(mic_pos)
		#assert not ((mic_pos >= _val2).any() or (mic_pos <= 0).any())


		
		sourceDataset = LibriSpeechDataset(_libri_speech_path, T, return_vad=True, size = 1)
		train_dataset = RandomTrajectoryDataset(sourceDataset, room_sz, T60, abs_weights, array_setup, array_pos, SNR, nb_points, transforms=[SRPDNN_features()])

		print(f"LibriSpeech: {len(sourceDataset)}, train_ds: {len(train_dataset)}")
		#mic_signals, dp_signals, acoustic_scene = train_dataset.__getitem__(3)
		#breakpoint()
		#print(f"inp_shape: {mic_signals.shape}, tgt_shape: {dp_signals.shape}")
		
		windowing = TorchChunking()
		from torch.utils.data import DataLoader
		train_loader = DataLoader(train_dataset, batch_size = 6)#, num_workers=2)

		for _batch_idx, val in enumerate(train_loader):
			print(f"inp_shape: {val[0].shape}, inp_dtype: {val[0].dtype}, tgt_shape: {val[1].shape}, tgt_dtype: {val[1].dtype}, doA: {val[2].shape}, doA_dtype: {val[2].dtype}")
			#print(val[2])
			#torch.save(val[2], 'doA_trajectory.pt')

			#doA_at_frm_level = windowing.get_frames(val[2])
			#print(f"doA_shape: {doA_at_frm_level.shape}, doA: {doA_at_frm_level}")
		
		#breakpoint()
		#import torchaudio

		#torchaudio.save("signal.wav", torch.from_numpy(mic_signals.T).to(dtype=torch.float32, device = "cpu"), 16000)
		#torchaudio.save("dp_signal.wav", torch.from_numpy(dp_signals.T).to(dtype=torch.float32, device="cpu"), 16000)


		#print(f"{mic_signals.shape}, {acoustic_scene.DOA}")
		#breakpoint()
	else:
		_locata_speech_path = '/fs/scratch/PAS0774/Shanmukh/Databases/Locata/eval/'
		train_dataset = LocataDataset(_locata_speech_path, 'benchmark2', 16000, [3], dev=True, transforms=[SRPDNN_Test_features(1,2)])
		from torch.utils.data import DataLoader
		train_loader = DataLoader(train_dataset, batch_size = 2)
		for _batch_idx, val in enumerate(train_loader):
			print(f"inp_shape: {val[0].shape}, inp_dtype: {val[0].dtype}, tgt_shape: {val[1].shape}, tgt_dtype: {val[1].dtype}, doA: {val[2].shape}, doA_dtype: {val[2].dtype}")
