class BankCard:
    def __init__(self, money):
        self.money = money
        # self.balance

    def __repr__(self):
        self.money -= 0
        return 'To learn the balance you should put the money on the card, spent some money or get the bank data. The last procedure is not free and costs 1 dollar.'

    def __call__(self, sum_spent):
        if sum_spent <= self.money:
            self.money -= sum_spent
            return print(f"You spent {sum_spent} dollars. {self.money} dollars are left.")
        else:

            raise ValueError(f'Not enough money to spent {sum_spent} dollars')

    def put(self, sum_put):
        self.money += sum_put
        return print(f"You put {sum_put} dollars. {self.money} dollars are left.")

    @property
    def total_sum(self):
        self.money -= 0
        return self.money

    @property
    def balance(self):
        if self.money - 1 < 0:
            raise ValueError(f' Not enough money to learn the balance')
        else:
            self.money -= 1
            return self.money