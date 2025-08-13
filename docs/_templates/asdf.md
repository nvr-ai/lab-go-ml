<%*
const newFileName = tp.date.now("YYYY-MMM-DD hh-mm-ss ZZ");
await tp.file.move("/Inbox/" + newFileName);
-%>
<% "---" %>
key: value
<% "---" %>

# <% newFileName %>

<% tp.file.cursor() %>
