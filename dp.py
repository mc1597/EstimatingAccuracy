class Dirichlet:
	def __init__(self, alpha, maxNoOfClusters):
		self.alpha = 1.0
		self.clusterMemberCounts = []
		self.freeClusters = []
		self.populatedClusters = []
		self.clusterIDs = []
		self.clusterUnNormalizedProbabilities = []
		self.currentNumberOfClusters = 0
		self.dirichletProcess(self.alpha, maxNoOfClusters)

	def dirichletProcess(self, alpha, maxNoOfClusters):
		self.alpha = alpha
		self.clusterMemberCounts = [0]* maxNoOfClusters
		for clusterID in range(maxNoOfClusters):
			self.freeClusters.append(clusterID)
		self.clusterIDs = [0] * maxNoOfClusters
		self.clusterUnNormalizedProbabilities = [0] * maxNoOfClusters

	def addMemberToCluster(self, clusterID):
		self.clusterMemberCounts[clusterID] += 1
		if self.clusterMemberCounts[clusterID] == 1:
			self.freeClusters.remove(clusterID)
			self.populatedClusters.append(clusterID)
	
	def removeMemberFromCluster(self, clusterID):
		self.clusterMemberCounts[clusterID] -= 1
		if self.clusterMemberCounts[clusterID] == 0:
			self.freeClusters.append(clusterID)
			self.populatedClusters.remove(clusterID)

	def computeClusterDistribution(self):
		self.currentNumberOfClusters = 0
		j = 0
		for i in range(len(self.populatedClusters)):
			cluster = self.populatedClusters[i]
			self.clusterIDs[self.currentNumberOfClusters] = cluster
			self.clusterUnNormalizedProbabilities[self.currentNumberOfClusters] = self.clusterMemberCounts[cluster]
			self.currentNumberOfClusters += 1
		if len(self.freeClusters) == 0:
			return self.currentNumberOfClusters
		self.clusterIDs[self.currentNumberOfClusters] = self.freeClusters[j]
		self.clusterUnNormalizedProbabilities[self.currentNumberOfClusters] = self.alpha
		self.currentNumberOfClusters += 1
		return self.currentNumberOfClusters

	def getClusterID(self, clusterPosition):
		return self.clusterIDs[clusterPosition]

	def getClusterUnnormalizedProbability(self, clusterPosition):
		return self.clusterUnNormalizedProbabilities[clusterPosition]



