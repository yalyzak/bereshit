class UserManager:
    def __init__(self):
        self.offenders = {}     # username → count
        self.banned_ips = set() # permanently banned IPs
        self.tolerateMax = 3

    def report_offense(self, username, ip):
        self.offenders[username] = self.offenders.get(username, 0) + 1
        if self.offenders[username] >= self.tolerateMax:
            self.banned_ips.add(ip)
            return True, self.offenders[username]  # should be kicked
        return False, self.offenders[username]

    def is_banned(self, ip):
        return ip in self.banned_ips
