# Hi Justin !

(*Not sure the readme is supposed to be the place for update, but it's the first thing we see on github so I find it efficient :D*)



#### Where we're at : Regression function 
I think the function execute_r_script (hazelbean/stats) should be different on a mac because
*cmd = 'C:\\Program Files\\R\\R-3.3.1\\bin\\Rscript.exe --vanilla --verbose ' + script_uri*
won't work on a mac.

#### Next steps
- Writing execute_r_script for Mac (or not?)
Probably just needs the equivalent of *cmd = 'C:\\Program Files\\R\\R-3.3.1\\bin\\Rscript.exe --vanilla --verbose ' + script_uri* for mac, but i've played around a little with my installation of R (for some reason doesn't work on command line), anyway I'm sure it's do-able but not obvious, so I was wondering if you think we'll stick with the Python-calling-R. Otherwise not worth it to do R compatibility adventure now.


- Alternative regression functions.
