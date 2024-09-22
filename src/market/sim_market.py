# TODO simulate https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_equation
import numpy as np
import matplotlib.pyplot as plt


def price_derivative_fun(S, mu, dt, sigma, rng):
    return mu * S * dt + sigma*S*rng.normal(0, np.sqrt(dt), size=S.shape)


def normalize(vec):
    return vec /np.linalg.norm(vec, 1)


def market_sim(init, exp_return, time, rng):
    dt = time[1] - time[0]
    prices = [init]
    dxdts= []
    sigma = 0.8
    for _ in time:
        dxdt =  price_derivative_fun(prices[-1], exp_return, dt, sigma, rng)
        new_values = prices[-1] + dt *dxdt
        norm_new_values = normalize(new_values)
        prices.append(norm_new_values)
        dxdts.append(dxdt)

    prices = np.array(prices)
    dxdts = np.array(dxdts)
    return prices, dxdts

if __name__ == '__main__':
    rng = np.random.default_rng(42)

    t = 100
    dt = 0.1

    time = np.arange(0, t, dt)
    exp_return = np.array([.1, 0.])


    start_price = normalize(np.array([5, 4]))
    prices, _ = market_sim(start_price, exp_return, time,)

    plt.title("Titel Preise")
    plt.plot(time, prices[:-1])
    plt.grid("on")
    plt.show()

    plt.title("Summe aller Titel")
    plt.plot(np.linalg.norm(prices, axis=-1), label='2-norm')
    plt.plot(np.sum(prices, axis=-1), label='sum-norm')
    plt.legend()
    plt.show()

    pass