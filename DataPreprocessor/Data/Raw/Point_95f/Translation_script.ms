--out_name = getSaveFileName caption:"Pose" types:".dat"
--out_file = createfile out_name

animationRange = interval 0 94
startFrame = 0
endFrame = 94
	
for frameNum in startFrame to endFrame do (
	sliderTime = (int(frameNum))
	myfile = createfile ("Pose" + (int(frameNum) as string) +".txt")
	rbm = render channels:#(#zDepth)
	for y=0 to 239 do
	(
		for x=0 to 319 do
		(
			print (getChannel rbm[x,y] #zDepth) to:myfile
		)
	)
	close myfile
)

