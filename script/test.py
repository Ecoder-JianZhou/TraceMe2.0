# preset the positional of components: can not be changed !!!!
x,   xc,  xp,  npp, tau, gpp, cue, bstau, senv, stas, spr, tas, pr = range(13)
abx, abxc, abxp, abtau, abstau = range(5)
class test:

    def __init__(self):
       self.x = x

    def pri(self):
        print(x)

test = test()
test.pri()