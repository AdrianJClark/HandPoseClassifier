--out_name = getSaveFileName caption:"Pose" types:".dat"
--out_file = createfile out_name

animationRange = interval 0 100
startFrame = 0
endFrame = 95
	
for frameNum = startFrame to endFrame do (
	sliderTime = (int(frameNum))
	myfile = createfile ("Pose" + (int(frameNum) as string) +".txt")
	rbm = render outputsize:[320,240] channels:#(#zDepth) vfb:off
	for y=0 to 239 do
	(
		for x=0 to 319 do
		(
			myArray = (getChannel rbm[x,y] #zDepth)
			format "%\n" myArray[1] to:myfile
			--print (getChannel rbm[x,y] #zDepth) to:myfile
		)
	)
	close myfile
)

