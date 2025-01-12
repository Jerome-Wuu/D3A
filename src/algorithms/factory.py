from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.dda import DDA
from algorithms.d3a import D3A

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA,
	'dda': DDA,
	'd3a': D3A,
}


def make_agent(obs_shape, action_shape, args, image_dir):
	return algorithm[args.algorithm](obs_shape, action_shape, args, image_dir)
