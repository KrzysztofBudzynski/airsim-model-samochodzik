import airsim
client = airsim.VehicleClient()
version = client.getServerVersion()
print(version)
client.simAddVehicle('klej', 'PhysXCar', airsim.Pose.nanPose)