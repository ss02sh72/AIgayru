import time

start_time = time.time()

a = "a"

def T():
    time.sleep(1)

def H(a):
    print("ひまり : " +  str(a))
    T()

def I(a):
    print("自分 : "  + str(a))
    T()

def N(a):
    print("ナレーション : " +  str(a))
    T()

def IN():
    print("会話を入力してください : ")
    global a
    a = input()
    T()

def YN():
    print("はい、か いいえ で答えてください")#最後修正
    IN()


point = 0

#入学式から初一緒行動

I("今日は待ちに待った大学の入学式だ。最高の大学生を過ごすぞ。")

N("現在、入学式中、隣に座った女の子が話しかけてきた")

H("私の名前は、佐藤ひまりよろしくね！")

H("隣になったのも何かの縁だしご飯でも食べいかない？")

YN()

for i in range(3):

    if a == "はい":
        H("じゃあご飯いこっか！")
        break

    elif a == "いいえ":
        H("じゃあ散歩行こっか！")
        break

    else:
        H("隣になったのも何かの縁だしご飯でも食べいかない？")
        print("はい、か いいえ を入力して下さい")
        IN()
        continue

if a == "はい":

    H("何食べいく食べに行く？")

    N("（近くにはカレー屋とラーメン屋と定食屋がある）")
    print("カレー か ラーメン か 定食 を選んでください")#最後修正
    IN()

    for i in range(3):
        c = i + 1
        if a == "カレー":
            H("カレー好きなんだ！楽しみ...!")
            point = point + 1
            print("二人はカレー屋さんについて席に案内された")
            break

        elif a == "ラーメン":
            H("いいね！ラーメン、レッツゴー！")
            print("二人はラーメン屋さんについて席に案内された")
            break
        
        elif a == "定食":
            H("あそこ行ってみたかったんだよね！いこうか！")
            print("二人は定食屋さんについて、席に案内された")
            break
        
        else:
            H("どこのご飯食べに行く？")
            N("（近くにはカレー屋とラーメン屋と定食屋がある）")
            print("カレー か ラーメン か 定食 を選んでください")#最後修正
            IN()
            continue
    if c == 3:
        point = point -1

else:
    N("ふたりは散歩を始めた")

T()

#ふたりの飯屋または散歩での会話

H("そういえば、君は出身はどこなの？")

IN()

if a == "茨城" or a == "茨城県":
    point = point + 2
    H("私も茨城　出身なんだ！一緒なんだね...!")

H("そうなんだね！趣味はとかはあるの？")

print("インドア派 か アウトドア派 かを選んでください")

IN()
for i in range(3):
    c = i + 1
    if a == "インドア派":
        print("そうなんだね！私は結構アウトア派なんだよねー")
        

        othello = 1
        break

    elif a == "アウトドア派":
        print("私もアウトドア派なんだ！私たち気が合うかもね!")
        point = point + 2
        othello = 0
        break

    else:
        H("趣味とかあるの？")
        print("インドア派 か アウトドア派 かを選んでください")
        IN()
        continue

if c == 3:
    H("まーいいや")
    point = point -1


H("好きな食べ物は何なの?")

print("和食 か 洋食 かを選んでください")

IN()
for i in range(3):
        
    if a == "和食":
        H("私は洋食のが好きだなー")
        break

    elif a =="洋食":
        H("私も洋食が好きなんだ！一緒だね！")
        point = point + 1
        break

    else:
        
        H("好きな食べ物は何なの?")
        print("和食 か 洋食 かを選んでください")
        IN()
        continue

if c == 3:
    H("まーいいや")
    point = point - 1

H("ところでさ、彼女とかはいたりするの...?")

N("ひまりはもじもじしながら聞いてきた。")

YN()

for i in range(2):
    c = i + 1
    if a == "はい":
        H("そうなんだ、モテそうだもんね...")
        point = point -1
        break
    elif a == "いいえ":
        H("えー、もてそうなのに意外だね！")
        break
    else:
        H("えーどっちなのー？？")
        N("ひまりはもじもじしながらもう一度聞いてきた。")
        IN()
        continue

if c == 2:
    H("ごめん、プライベートすぎること聞いちゃったね...")
    point = point + 2

if othello == 0:
    H("ところでなんだけどさ、私ね、剣道サークルに興味あるんだよね！、一緒に行こうよ！！")
    N("こうして二人はサークルの体験に行くことになった")
    T()

else:
    H("ところでなんだけどさ、私ね、オセロサークルに興味あるんだよね！、一緒に行こうよ！！よ")
    N("こうして二人はサークルの体験に行くことになった")
    T()



#サークル体験
if othello == 1:
    N("二人は一緒にサークル見学に来ました 二人が来たのはボードゲームサークルです")
    H("大学のサークルってどんな感じなんだろうね？")
    H("楽しみだけどちょっと緊張するね")
    I("陽葵もこういう場は緊張するんだね そんな感じしないけど")
    H("大学の初めてのサークル見学だよ？ めちゃめちゃ緊張するよ")
    H("でも君と一緒だからちょっとは緊張しなくなってる")
    N("二人はいろいろあるボードゲームの中からオセロをやることにしました")
    H("私こういう頭使う系のゲームあんまり普段やんないんだよねー")
    H("君は普段オセロかやったりするの？")
    YN()
    for i in range(3):
        c = i + 1
        if a == "はい":
            H("すごーい 頭使うの好きなんだね ちょっと尊敬")
            H("でも負けないよー！！私地頭はいいからね（笑）")
            N("二人はオセロの対戦を楽しみました")
            time.sleep(3)
            H("うわーー！！負けたーー！！")
            H("さすが普段やってるだけあるね！")
            H("やっぱ思考速度の違いを感じるわ")
            break
        elif a == "いいえ":
            H("私もー ボードゲームとかあんましやる機会ないよね")
            H("お互い初心者ってことか")
            H("私加減とかしないから 君も全力で来てね")
            time.sleep(3)
            H("うわーー！！負けたーー！！")
            H("結構接戦だった感あったんだけど")
            H("次やったら負けないよ")
            break
        else:
            H("君は普段オセロかやったりするの？")
            YN()
            continue
    if c == 3:
        point -= 1
        H("まぁいっか とりあえず始めよ")
        time.sleep(3)
        H("うわーー！！負けたーー！！")
        H("結構接戦だった感あったんだけど")
        H("次やったら負けないよ")
    else:
        pass
    H("今まで話して手の思ってたけど")
    H("君って結構頭いいよね？")
    print("[はい],[いいえ],[分からない]で答えてください")
    IN()
    for i in range(3):
        c = i + 1
        if a == "はい":
            H("やっぱりそうだよね")
            break
        elif a == "いいえ":
            point += 1
            H("君は私をオセロで負かした")
            H("頭いいよ")
            H("やっぱり頭よかったらさ...")
            break
        elif a == "分からない":
            point += 2
            H("頭のよさって相対評価だしね")
            H("謙虚だね...")
            H("やっぱり頭よかったらさ...")
        else:
            H("君って結構頭いいよね？")
            continue
    if c == 3:
        point -= 1
        H("まーいっか やっぱ君頭いいよ")
        H("やっぱり頭よかったらさ...")
if othello == 0:
    N("二人は二人はアットホームな雰囲気にひかれて剣道サークルにやってきました")
    H("大学のサークルってどんな感じなんだろうね？")
    H("楽しみだけどちょっと緊張するね")
    I("陽葵もこういう場は緊張するんだね そんな感じしないけど")
    H("大学の初めてのサークル見学だよ？ めちゃめちゃ緊張するよ")
    H("でも君と一緒だからちょっとは緊張しなくなってる")
    H("私は中高で剣道やってたんだよね")
    H("君は剣道やってたの？")
    YN()
    for i in range(3):
        c = i + 1
        if a == "はい":
            H("え？まじで？一緒じゃん！！")
            H("でも負けないよー！！真剣勝負！！")
            N("二人は剣道の試合をしました 真剣勝負！！")
            time.sleep(3)
            H("うわーー！！負けたーー！！")
            H("一回目の面はすごかったね！")
            H("完敗だよー")
            break
        elif a == "いいえ":
            H("へー でも道着姿けっこう様になってるよ")
            H("いろいろ教えてあげるから 楽しくやろうね")
            time.sleep(3)
            H("うわーー！！負けたーー！！まじかーー！！")
            H("さすがに勝てると思ったよー")
            H("次やったら負けないよ")
            break
        else:
            H("君は剣道やってたの？")
            YN()
            continue
    if c == 3:
        point -= 1
        H("まぁいっか とりあえず始めよ")
        time.sleep(3)
        H("うわーー！！負けたーー！！")
        H("結構接戦だった感あったんだけど")
        H("次やったら負けないよ")
    H("今日練習して思ったけど...")
    H("君って運動神経めっちゃいいよね？")
    print("[はい],[いいえ],[分からない]で答えてください")
    IN()
    for i in range(3):
        c = i + 1
        if a == "はい":
            H("やっぱりそうだよね")
            H("やっぱり運動できるとさ...")
            break
        elif a == "いいえ":
            point += 1
            H("うそー 動きめっちゃ俊敏だったよ")
            H("すごかった")
            H("やっぱり運動できるとさ...")
            break
        elif a == "分からない":
            point += 2
            H("まあ上には上がいるよね")
            H("謙虚だね...")
            H("やっぱり運動できるとさ...")
            break
        else:
            H("君って結構頭いいよね？")
            continue
    if c == 3:
        point -= 1
        H("まーいっか やっぱ君運動神経いいよ")
        H("やっぱり運動できるとさ...")
H("モテるよね")
print("[はい],[いいえ],[分からない]で答えてください")
IN()
while True:
    if a == "はい":
        point += 0
        H("へー...")
        break
    elif a == "いいえ":
        point += 1
        H("へー意外だね")
        break
    elif a == "分からない":
        point += 3
        H("もーはぐらかさないでよお")
        break
    else:
        H("それはどっちなの？")
        print("[はい],[いいえ],[分からない]で答えてください")
        IN()
H("...,")
H("そういえばさ")
H("６月にこのサークルの本新歓あるらしいよ")
H("なんか地方の遊園地行くらしい")
H("なんか先輩は「ほどよくしょっぱい」？とか言ってた")
H("一緒に行こうよ")
N("二人はこのサークルに入ることにしました")
time.sleep(3)
N("本新歓当日")





#本新歓＠那須ハイランドパーク

H("とりあえず、なに乗ろうか？")

N("近くにはジェットコースターとコーヒーカップがある")

IN()

for i in range(3):
    if a == "ジェットコースター":
        point = point + 1
        N("二人はジェットコースターに乗って楽しんだ")
        break
    elif a == "コーヒーカップ":
        if point > 10:
            N("ふたりはコーヒーカップに乗った。しかし自分は体調が悪くなってしまい、朝ごはんを戻してしまった")
            H("大丈夫？？お水買ってきたから飲んで！！ほんとに大丈夫？？")
            N("ひまりの看病のおかげで気分はだんだんと良くなった。自分は普段よりも元気になった気がした")
            point = point + 3
            break
        else:
            N("ふたりはコーヒーカップに乗って楽しんだ")
            break
    else:
        H("とりあえず、なに乗ろうか？")

        N("近くにはジェットコースターとコーヒーカップがある")

        IN()

        continue




if c == 3:
    H("とりあえずジェットコースター乗ろうか！")
    point = point - 1

N("楽しい時間は流れ、お昼ご飯の時間帯になった")

H("結構一通り楽しんだね！おなかぺこぺこー何食べよっか？")
    
N("近くには やきとり、パスタ、たこ焼きが食べられる屋台がある。どれにするか選んでください")

IN()

for i in range(3):
    c = i + 1
    if a == "やきとり":
        N("二人はおいしくやきとりを食べた")
        break
    elif a == "パスタ":
        H("パスタめっちゃおいしいね！")
        N("ふたりは仲良くパスタをほおばった")
        point = point + 2
        break

    elif a == "たこ焼き":
        N("ふたりはなかよくたこ焼きをほおばった")
        point = point + 1
        break
    else:
        H("おなかぺこぺこー何食べよっか？")
    
        N("近くには やきとり、パスタ、たこ焼きが食べられる屋台がある。どれにするか選んでください")

        IN()

        continue
if c ==3:
    H("とりあえず近くのたこ焼き食べようか！")
    N("ふたりはおいしくたこ焼きを食べた")

T()

H("最後にふたりで観覧車乗ろうよ！！いこいこーーー！！")

N("ふたりは観覧車に乗ることになった")

T()

N("ふたりは観覧車から見える素晴らしい景色を見ながらゆっくりと進む時間を楽しんだ")

N("しかし、、、、")

N("ここで事件は起こった")

N("観覧車が途中で止まってしまった")

N("アナウンスが流れ、その内容は、、観覧車が故障したため、現在停止しているということだった")

N("ひまりは恐怖のあまり震えだしてしまうほど、おびえていた")

N("ここで、一緒にいるのはあなただけです。ひまりに慰めの言葉をかけてあげるのです！！")

N("次の選択肢から選ぶのです！！")

N("1. 大丈夫僕がついてるよ　2.観覧車なんてそうそう落ちるものじゃないから大丈夫だよ　3.なんでそんなおびえてるの")

N("１から３の選択肢から選んでください")

IN()

a = int(a)

for i in range(3):
    c = i + 1
    if a == 1:
        N("声をかけるとひなみの震えは止まり、安心したような顔をしていた")
        point = point + 4
        break

    elif a == 2:
        N("ひなみは少し安心したような様子だ")
        break
    elif a == 3:
        N("ひなみの様子は変わらなかった") 
        point = point -3
        break

    else:
        N("1. 大丈夫僕がついてるよ　2.観覧車なんてそうそう落ちるものじゃないから大丈夫だよ　3.なんでそんなおびえてるの")

        N("１から３の選択肢から選んでください")

        IN()

        continue

if c == 3:
    N("ひなみの様子は変わらなかった") 
    point = point -3

N("アナウンスとともに観覧車は動き出し、ふたりは安どした")

H("ひとときはどうなるかと思ったけど、私たち生きてる！！君がついていてくれたおかげだね！")

N("色んなことがあったが楽しい一日となった。")

rate = 100*point/22


N("あなたの得点率は、")
N(str(rate))
N("でした。")
N("いったんお疲れさまでした")
N("得点は、")
N(str(point))


end_time = time.time()


execution_time_1 = end_time - start_time

print(f"実行時間: {execution_time_1} 秒")