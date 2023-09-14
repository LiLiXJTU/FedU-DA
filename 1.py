xm_vote = int(input("请输入小明部门同事投票数："))
xm_yeji = int(input("请输入小明的去年业绩："))
xm_age = int(input("请输入小明的年龄："))
xm_gl = int(input("请输入小明的工龄："))
xm_yanj = int(input("请输入小明演讲得分："))
xq_vote = int(input("请输入小强部门同事投票数："))
xq_yeji = int(input("请输入小强的去年业绩："))
xq_age = int(input("请输入小强的年龄："))
xq_gl = int(input("请输入小强的工龄："))
xq_yanj = int(input("请输入小强演讲得分："))
xm_zf = xm_vote*3+(xm_yeji//20000)+(xm_gl-xq_gl)*5+xm_yanj
xq_zf = xq_vote*3+(xq_yeji//20000)+(xm_age-xq_age)*2+xq_yanj
print(xm_zf>xq_zf)