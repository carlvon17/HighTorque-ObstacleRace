cmake_minimum_required(VERSION 2.8.3)

project(PiPlusPro_SF_12L8A0G2H0W_Z_261010)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

foreach(dir config launch meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
